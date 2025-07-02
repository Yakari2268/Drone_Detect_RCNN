import torch
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import functional as F
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import sys

# Define class names
CLASS_NAMES = ['background', 'drone']

# Load model
model_path = "trained_models/drone_detect_RCNN.pt"

model = torch.load(model_path, map_location="cpu",weights_only=False)
model.eval()

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

## CSRT: --- State Management Variables ---
# We need variables to manage whether we are in "detection" or "tracking" mode.
tracker = None
tracking_mode = False
# ----------------------------------------

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- We now have two modes: Detection and Tracking ---
    
    ## CSRT: If we are in TRACKING mode
    if tracking_mode:
        # Update the tracker
        success, box = tracker.update(frame)

        # Draw the bounding box
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking Failed!", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display instructions for tracking mode
        cv2.putText(frame, "Press 'r' to reset detector", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    ## CSRT: If we are in DETECTION mode
    else:
        # This is your original detection code
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(image_rgb).to(device)

        with torch.no_grad():
            prediction = model([img_tensor])[0]

        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        
        # Find the best 'drone' detection to be our potential target
        best_score = 0
        potential_bbox = None
        
        for box, label, score in zip(boxes, labels, scores):
            # We are only interested in drones (label 1) with high confidence
            if label == 1 and score > 0.7:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(frame, f"Drone: {score:.2f}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Keep track of the detection with the highest score
                if score > best_score:
                    best_score = score
                    # CSRT needs bbox in (x, y, w, h) format
                    x1, y1, x2, y2 = map(int, box)
                    potential_bbox = (x1, y1, x2 - x1, y2 - y1)
        
        # Store the best potential box to be used if 's' is pressed
        bbox_to_track = potential_bbox

        # Display instructions for detection mode
        if bbox_to_track:
            cv2.putText(frame, "Press 's' to start tracking the best target", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Detecting...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --- Display the final frame ---
    cv2.putText(frame, "Press 'q' to quit", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Drone Detection and Tracking", frame)

    # --- Handle Key Presses ---
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    
    # 's' key to start tracking
    elif key == ord('s') and not tracking_mode and bbox_to_track is not None:
        print("Starting tracker...")
        # Initialize the CSRT tracker
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox_to_track)
        tracking_mode = True # Switch to tracking mode

    # 'r' key to reset to detection mode
    elif key == ord('r') and tracking_mode:
        print("Resetting to detection mode...")
        tracker = None
        tracking_mode = False

cap.release()
cv2.destroyAllWindows()