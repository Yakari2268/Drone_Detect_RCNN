import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import ToTensor
from src.dataclass_implementation import CustomVOCDataset  # Change to your actual class import

# Set your paths and classes
images_dir = "C:/Users/abhia/Documents/Iot/dataset/DUT-Anti-UAV/detection/train/images"
annotations_dir = "C:/Users/abhia/Documents/Iot/dataset/DUT-Anti-UAV/detection/train/annotations"
classes = ['__background__', 'UAV']  # <-- your label set

# Load dataset without DataLoader
dataset = CustomVOCDataset(images_dir, annotations_dir, classes, transforms=ToTensor())

# Function to visualize one sample
def visualize_sample(index):
    image, target = dataset[index]
    image_np = image.permute(1, 2, 0).numpy()  # CHW -> HWC

    fig, ax = plt.subplots(1)
    ax.imshow(image_np)

    boxes = target["boxes"]
    labels = target["labels"]

    if boxes.numel() == 0:
        print(f"[⚠️] Image {index} has no valid boxes.")
        return

    for i in range(len(boxes)):
        x_min, y_min, x_max, y_max = boxes[i].tolist()
        label_id = labels[i].item()
        label_name = classes[label_id] if label_id < len(classes) else "Unknown"
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, label_name, color='white', bbox=dict(facecolor='red', alpha=0.5))
    plt.show()

# Loop over samples to check
for i in range(len(dataset)):
    try:
        print(f"\n✅ Checking sample {i}")
        visualize_sample(i)
    except Exception as e:
        print(f"❌ Error in sample {i}: {e}")
