import torch
from torchvision.transforms import functional as F
import cv2
import numpy as np
import sys
import os
import re
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = "trained_models/drone_detect_RCNN.pt"
REID_MODEL_PATH = "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
DEEPSORT_CONFIG_PATH = "deep_sort_pytorch/configs/deep_sort.yaml"
CONF_THRESHOLD = 0.5
TRACK_CLASS_ID = 1
# --- END CONFIGURATION ---

# --- SETUP PATHS AND IMPORTS ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    deepsort_path = os.path.join(script_dir, 'deep_sort_pytorch')
    sys.path.append(deepsort_path)
    from deep_sort.deep_sort import DeepSort
    from utils.parser import get_config
except ImportError as e:
    print(f"Error importing DeepSORT: {e}")
    print("Please make sure the 'deep_sort_pytorch' directory is in the same folder as this script.")
    sys.exit(1)
# --- END SETUP ---

# --- MODEL INITIALIZATION ---
print("Loading detection model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Detection model not found at: {MODEL_PATH}")
CLASS_NAMES = ['background', 'drone']
model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)
# --- END MODEL INITIALIZATION ---

# --- DEEPSORT INITIALIZATION ---
print("Initializing DeepSORT...")
cfg_deep = get_config()
cfg_deep.merge_from_file(DEEPSORT_CONFIG_PATH)
if not os.path.exists(REID_MODEL_PATH):
    raise FileNotFoundError(f"Re-ID model for DeepSORT not found at: {REID_MODEL_PATH}")
deepsort = DeepSort(
    REID_MODEL_PATH,
    max_dist=cfg_deep.DEEPSORT.MAX_DIST,
    min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
    nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
    max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
    max_age=cfg_deep.DEEPSORT.MAX_AGE,
    n_init=cfg_deep.DEEPSORT.N_INIT,
    nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
    use_cuda=torch.cuda.is_available()
)
# --- END DEEPSORT INITIALIZATION ---

# --- VIDEO AND VISUALS SETUP ---
cap = cv2.VideoCapture(0)
np.random.seed(42) 
colors = np.random.randint(0, 255, size=(200, 3), dtype=np.uint8)
# --- END VIDEO SETUP ---

print("\nStarting video stream... Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # 1. --- DETECTION ---
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(img_tensor)[0]
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    # 2. --- ROBUST DETECTION FORMATTING ---
    detections_for_tracker = []
    indices = np.where((labels == TRACK_CLASS_ID) & (scores > CONF_THRESHOLD))[0]
    for i in indices:
        bbox = boxes[i]
        score = scores[i]
        label = labels[i]
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        detections_for_tracker.append([x1, y1, w, h, score, label])

    # 3. --- UPDATE TRACKER ---
    if len(detections_for_tracker) > 0:
        detections_np = np.array(detections_for_tracker)
        xywhs = torch.from_numpy(detections_np[:, :4]).float()
        confss = torch.from_numpy(detections_np[:, 4]).float()
        clss = torch.from_numpy(detections_np[:, 5]).float()
        outputs = deepsort.update(xywhs, confss, clss, frame)
    else:
        deepsort.update(torch.empty(0, 4), torch.empty(0), torch.empty(0), frame)
        outputs = torch.empty(0, 6)

    # 4. --- VISUALIZE TRACKING RESULTS ---
    frame_with_boxes = frame.copy()
    
    # --- NEW DIAGNOSTIC PRINTS ---
    print("\n--- FRAME START ---")
    print(f"[DEBUG] Tracker returned `outputs` of type: {type(outputs)}, with length/size: {len(outputs) if hasattr(outputs, '__len__') else 'N/A'}")
    if hasattr(outputs, 'shape'):
        print(f"[DEBUG] Shape of `outputs`: {outputs.shape}")
    # --- END DIAGNOSTIC PRINTS ---

    if len(outputs) > 0:
        print(f"[DEBUG] Raw content of `outputs`:\n{outputs}") # Print the raw content if not empty
        for output in outputs:
            # --- MORE DIAGNOSTIC PRINTS ---
            output_len = 'N/A'
            if hasattr(output, '__len__'):
                output_len = len(output)
            print(f"  -> [DEBUG] Processing track element `output` of type: {type(output)}, with length: {output_len}")
            # --- END DIAGNOSTIC PRINTS ---

            try:
                x1, y1, x2, y2, class_id, track_id = map(int, output)
                print(f"    --> [SUCCESS] Unpacked Track ID: {track_id}") # Will only print if unpacking works
            except (TypeError, ValueError):
                print(f"    --> [WARNING] Malformed tracker output skipped: {output}")
                continue
            
            color = colors[track_id % len(colors)].tolist()
            label_text = f"Drone ID: {track_id}"
            
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_with_boxes, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(frame_with_boxes, "Status: Detecting...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    cv2.imshow("Real-Time Drone Tracking", frame_with_boxes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("Stream finished.")