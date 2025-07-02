import os
import time
import torch
import cv2
import numpy as np
from YOLOv5 import letterbox, scale_coords
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F
import sys

from sort import sort

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

mot_tracker = sort.Sort(min_hits=3, max_age=2, iou_threshold=0.3)

model_path = "./trained_models/drone_detect_RCNN.pt"
CLASS_NAMES = ['background', 'drone']
font = ImageFont.load_default()

def load_model():
    device = select_device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_1 = DetectMultiBackend('yolov5s.pt', device=device)
    model_2 = torch.load(model_path, map_location=device, weights_only=False)
    model_2.eval()
    return model_1, model_2

def detect_from_folder(model1, model2, class_names, device, folder_path, frame_rate=30):
    frame_delay = 1.0 / frame_rate
    stride = model1.stride
    img_size = 640
    names = model1.names

    frame_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])

    for frame_path in frame_files:
        start_time = time.time()

        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        base_frame = frame.copy()

        # YOLOv5
        img = letterbox(base_frame, img_size, stride=stride)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img_tensor1 = torch.from_numpy(img).to(device).float() / 255.0
        if img_tensor1.ndimension() == 3:
            img_tensor1 = img_tensor1.unsqueeze(0)

        pred1 = model1(img_tensor1)
        pred1 = non_max_suppression(pred1, 0.25, 0.45)

        for det in pred1:
            if len(det):
                det[:, :4] = scale_coords(img_tensor1.shape[2:], det[:, :4], base_frame.shape).round()
                for *xyxy, conf, cls in det:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(base_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(base_frame, label, (int(xyxy[0]), int(xyxy[1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # RCNN
        image_pil = Image.fromarray(cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB))
        img_tensor2 = F.to_tensor(image_pil).to(device)

        with torch.no_grad():
            prediction = model2([img_tensor2])[0]

        draw = ImageDraw.Draw(image_pil)
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        # RCNN drone detections
        drone_detections = []
        for box, label, score in zip(boxes, labels, scores):
            if score >= 0.5 and label == 1:
                drone_detections.append([*box, score])

        drone_detections_np = np.array(drone_detections)

        if drone_detections_np.shape[0] > 0:
            tracked_objects = mot_tracker.update(drone_detections_np)
        else:
            tracked_objects = mot_tracker.update(np.empty((0, 5)))

        # Draw RCNN boxes
        for box, label, score in zip(boxes, labels, scores):
            if score < 0.5:
                continue
            box = list(map(int, box))
            class_name = class_names[label] if label < len(class_names) else str(label)
            text = f"{class_name}: {score:.2f}"
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1] - 10), text, fill="red", font=font)

        # Draw tracked drones
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
            draw.text((x1, y1 - 15), f"Drone ID: {int(track_id)}", fill="blue", font=font)

        # Show frame
        final_frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow("Drone Detection (Simulated Real-time)", final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Frame rate control
        elapsed = time.time() - start_time
        time.sleep(max(0, frame_delay - elapsed))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1, model2 = load_model()
    
    # 🔁 Simulate real-time playback from a folder:
    FRAME_FOLDER = "C:/Users/abhia/Downloads/Anti-UAV-Tracking-V0/Anti-UAV-Tracking-V0/video01"
    detect_from_folder(model1, model2, CLASS_NAMES, device, FRAME_FOLDER, frame_rate=60)
