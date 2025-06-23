from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
import src.dataclass_implementation as dataclass_implementation
import torchvision.transforms as T

# Define your class list (must include '__background__' at index 0)
classes = ['__background__', 'UAV']

# Optional: define transforms
transform = T.Compose([
    T.ToTensor(),  # Converts PIL image to FloatTensor and scales [0, 255] to [0.0, 1.0]
])

# Initialize dataset and dataloader
dataset = dataclass_implementation.CustomVOCDataset('C:/Users/abhia/Documents/Iot/dataset/DUT-Anti-UAV/detection/train/images', 'C:/Users/abhia/Documents/Iot/dataset/DUT-Anti-UAV/detection/train/annotations', classes, transforms=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Load model
model = fasterrcnn_resnet50_fpn(num_classes=len(classes))

# Sample batch
for imgs, targets in dataloader:
    output = model(list(imgs), list(targets))  # During training
    break
