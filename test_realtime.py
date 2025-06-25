import torch
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import functional as F
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Define class names
CLASS_NAMES = ['background', 'drone']

# Load model
model_path = "drone_detect_RCNN.pt"
model = torch.load(model_path, map_location="cpu",weights_only=False)
model.eval()
torch.save(model.state_dict(), "drone_detect_RCNN_sd.pt")

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Load font
try:
    font = ImageFont.truetype("arial.ttf", 20)
except IOError:
    font = ImageFont.load_default()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert OpenCV BGR to PIL RGB
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Convert PIL image to tensor
    img_tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    draw = ImageDraw.Draw(image)

    for box, label, score in zip(boxes, labels, scores):
        if score < 0.5:
            continue
        box = box.tolist()
        class_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else str(label)
        text = f"{class_name}: {score:.2f}"
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1] - 10), text, fill="red", font=font)

    # Convert PIL image back to OpenCV BGR for display
    frame_with_boxes = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imshow("Real-Time Detection", frame_with_boxes)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
