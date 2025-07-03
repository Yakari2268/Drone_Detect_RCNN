import torch
import sys
import cv2
import numpy as np
from YOLOv5 import letterbox, scale_coords
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F

sys.path.insert(0, './yolov5')  # Adjust if yolov5 repo path differs

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

# -----------------------------
# CONFIG
# -----------------------------

# RCNN classes
RCNN_CLASS_NAMES = ['background', 'drone']

# Custom YOLO classes to detect
YOLO_CLASSES = ['person', 'airplane', 'bird']

# RCNN model path
RCNN_MODEL_PATH = "trained_models/drone_detect_RCNN.pt"

# YOLO model path (or use 'yolov5s.pt')
YOLO_MODEL_PATH = 'yolov5s.pt'

font = ImageFont.load_default()

# -----------------------------
# Load Models
# -----------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load RCNN model
rcnn_model = torch.load(RCNN_MODEL_PATH, map_location=device, weights_only=False)
rcnn_model.eval()

# Load YOLO model
device_yolo = select_device('0')
yolo_model = DetectMultiBackend(YOLO_MODEL_PATH, device=device_yolo)
stride = yolo_model.stride
yolo_imgsz = 640
yolo_names = yolo_model.names

# -----------------------------
# Initialize Webcam
# -----------------------------

cap = cv2.VideoCapture(0)

# -----------------------------
# CSRT Tracking State
# -----------------------------

tracker = None
tracking_mode = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_disp = frame.copy()

    # -----------------------------
    # Tracking Mode
    # -----------------------------
    if tracking_mode:
        success, box = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame_disp, (x, y), (x + w, y + h), (225, 0, 0), 2)
            cv2.putText(frame_disp, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame_disp, "Tracking Failed!", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame_disp, "Press 'r' to reset detector", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # -----------------------------
    # Detection Mode
    # -----------------------------
    else:
        bbox_to_track = None
        best_score = 0

        # --- YOLO Detection ---
        yolo_img = letterbox(frame, yolo_imgsz, stride=stride)[0]
        yolo_img = yolo_img.transpose((2, 0, 1))[::-1]  # BGR to RGB, HWC to CHW
        yolo_img = np.ascontiguousarray(yolo_img)

        img_tensor = torch.from_numpy(yolo_img).to(device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        pred = yolo_model(img_tensor)
        pred = non_max_suppression(pred, 0.25, 0.45)

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    cls_name = yolo_names[int(cls)]
                    if cls_name not in YOLO_CLASSES:
                        continue

                    label = f"{cls_name} {conf:.2f}"
                    cv2.rectangle(frame_disp,
                                  (int(xyxy[0]), int(xyxy[1])),
                                  (int(xyxy[2]), int(xyxy[3])),
                                  (0, 255, 0), 2)
                    cv2.putText(frame_disp, label,
                                (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)

        # --- RCNN Detection ---
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor_rcnn = F.to_tensor(image_rgb).to(device)

        with torch.no_grad():
            prediction = rcnn_model([img_tensor_rcnn])[0]

        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if label == 1 and score > 0.7:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_disp, f"Drone: {score:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)

                if score > best_score:
                    best_score = score
                    bbox_to_track = (x1, y1, x2 - x1, y2 - y1)

        if bbox_to_track:
            cv2.putText(frame_disp, "Press 's' to start tracking the best target",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
        else:
            cv2.putText(frame_disp, "Detecting...", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

    # Display info
    cv2.putText(frame_disp, "Press 'q' to quit", (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("YOLO + RCNN + Tracking", frame_disp)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    elif key == ord('s') and not tracking_mode and bbox_to_track is not None:
        print("Starting tracker...")
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox_to_track)
        tracking_mode = True

    elif key == ord('r') and tracking_mode:
        print("Resetting to detection mode...")
        tracker = None
        tracking_mode = False

cap.release()
cv2.destroyAllWindows()
