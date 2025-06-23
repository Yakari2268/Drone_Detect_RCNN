import torch
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt

# Define your class names for the dataset
CLASS_NAMES = ['background', 'drone']

# Load model with safe_globals context
model_path = "best.pt"
with torch.serialization.safe_globals([FasterRCNN]):
    model = torch.load(model_path, weights_only=False)
model.eval()

# Device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Directory of test images
test_dir = "C:/Users/abhia/Documents/Iot/dataset/DUT-Anti-UAV/detection/test/img"

# Directory to save results
output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)  # create folder if not exists

for img_name in os.listdir(test_dir):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue

    img_path = os.path.join(test_dir, img_name)
    image = Image.open(img_path).convert("RGB")

    img_tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        predictions = model([img_tensor])

    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        if score < 0.5:
            continue
        box = box.tolist()

        # Draw bounding box
        draw.rectangle(box, outline="red", width=2)

        # Use your custom class names
        class_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else str(label)
        text = f"{class_name}: {score:.2f}"

        # Draw label text
        draw.text((box[0], box[1]), text, fill="red", font=font)

    # Save the image with boxes to output directory
    save_path = os.path.join(output_dir, img_name)
    image.save(save_path)

    # Optional: display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title(f"Predictions for {img_name}")
    plt.axis('off')
    plt.show()
