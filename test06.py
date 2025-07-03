import torch
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import functional as F
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import sys
import os

## DS: Add the deep_sort_pytorch directory to the Python path
# This allows us to import the deep_sort library
sys.path.append(os.path.join(os.path.dirname(__file__), 'deep_sort_pytorch'))

## DS: Import DeepSORT
from deep_sort.deep_sort import DeepSort
from utils.parser import get_config

# Define class names
CLASS_NAMES = ['background', 'drone']

# Load model
model_path = "trained_models/drone_detect_RCNN.pt"
model = torch.load(model_path, map_location="cpu", weights_only=False)
model.eval()

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

## DS: --- Initialize DeepSORT ---
# We need to define the configuration file for DeepSORT and the path to the Re-ID model.
cfg_deep = get_config()
cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

# Make sure the Re-ID model path is correct
reid_model_path = "./trained_models/checkpoints/ckpt.t7"
if not os.path.exists(reid_model_path):
    raise FileNotFoundError(f"Re-ID model not found at {reid_model_path}. Please ensure the deep_sort_pytorch repo is structured correctly.")

deepsort = DeepSort(
    reid_model_path,
    max_dist=cfg_deep.DEEPSORT.MAX_DIST,
    min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
    nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
    max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
    max_age=cfg_deep.DEEPSORT.MAX_AGE,
    n_init=cfg_deep.DEEPSORT.N_INIT,
    nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
    use_cuda=True  # Make sure this matches your device
)
## DS: -------------------------

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Load font
try:
    font = ImageFont.truetype("arial.ttf", 20)
except IOError:
    font = ImageFont.load_default()

## DS: Create a color map for different track IDs
np.random.seed(42) # for consistent colors
colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. --- DETECTION ---
    # Convert OpenCV BGR to PIL RGB
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    # 2. --- FORMAT DETECTIONS FOR DEEPSORT ---
    # We need to convert the detections from our model into the format DeepSORT expects:
    # A list of tuples, where each tuple is ([x1, y1, w, h], confidence, class_id)
    
    # Filter out low-confidence detections and detections that are not 'drone'
    # Assuming 'drone' is class 1

    detections_for_tracker = []
    drone_indices = (labels == 1) & (scores > 0.5)
    
    bboxes_xywh = []
    confs = []
    clss = []
    
    for i in np.where(drone_indices)[0]:
        bbox = boxes[i]
        score = scores[i]
        label = labels[i]

        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        bboxes_xywh.append([x1, y1, w, h])
        confs.append(score)
        clss.append(label)

        detections_for_tracker.append([x1,y1,w,h,score,label])

    # 3. --- UPDATE TRACKER ---
    xywhs = torch.tensor(bboxes_xywh)
    confss = torch.tensor(confs)
    clss_tensor = torch.tensor(clss)

    if len(detections_for_tracker) > 0:
        # Convert the list to a NumPy array for robust slicing
        detections_np = np.array(detections_for_tracker)

        # Slice the NumPy array to get the required components
        # This correctly handles the case of a single detection, keeping it as a 2D array
        xywhs_np = detections_np[:, :4]
        confs_np = detections_np[:, 4]
        clss_np = detections_np[:, 5]

        # Convert the NumPy arrays to PyTorch tensors
        xywhs = torch.from_numpy(xywhs_np).float()
        confss = torch.from_numpy(confs_np).float()
        clss_tensor = torch.from_numpy(clss_np).float()
        
        # Pass the correctly shaped tensors to the tracker
        outputs = deepsort.update(xywhs, confss, clss_tensor, frame)

    else:
        # If no detections, call update with empty tensors
        # This is crucial for the tracker to age and manage existing tracks
        deepsort.update(torch.empty(0, 4), torch.empty(0), torch.empty(0), frame)
        outputs = torch.empty(0, 5) # Ensure outputs is an empty tensor

    # 4. --- VISUALIZE TRACKING RESULTS ---
    # The `outputs` variable now contains tracked objects with their IDs
    frame_with_boxes = frame.copy() # Work on a copy of the frame

    if len(outputs) > 0:
        for output in outputs:
            # `output` format: [x1, y1, x2, y2, track_id]

            if len(output) < 5:
                continue
            x1, y1, x2, y2, track_id = map(int, output)
            
            # Assign a unique color to each track ID
            color = colors[track_id % len(colors)].tolist()
            
            # Draw the bounding box
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
            
            # Create the label text with the track ID
            text = f"Drone ID: {track_id}"
            
            # Put the text above the bounding box
            cv2.putText(frame_with_boxes, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Real-Time Drone Tracking", frame_with_boxes)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()