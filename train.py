import os
import yaml
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import ToTensor
from src.dataclass_implementation import CustomVOCDataset  # <-- your dataset class
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ----------- Load Config -----------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ----------- Device Setup -----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
# ----------- Dataset & DataLoader -----------
classes = ['__background__', 'UAV']  # must match dataset labels
train_dataset = CustomVOCDataset(
    images_dir="C:/Users/abhia/Documents/Iot/dataset/DUT-Anti-UAV/detection/train/images",
    annotations_dir="C:/Users/abhia/Documents/Iot/dataset/DUT-Anti-UAV/detection/train/annotations",
    classes=classes,
    transforms=ToTensor()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["BATCH_SIZE"],
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

# ----------- Model Initialization -----------
model = fasterrcnn_resnet50_fpn_v2(
    weights='DEFAULT',
)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=len(classes))
model.to(device)

# ----------- Optimizer Setup -----------
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=config["LEARNING_RATE"],
    momentum=config["MOMENTUM"],
    weight_decay=config["WEIGHT_DECAY"]
)

# ----------- Training Loop -----------
for epoch in range(config["EPOCHS"]):
    print(f"\nEpoch {epoch+1}/{config['EPOCHS']}")
    model.train()
    epoch_loss = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
        print(f"[{batch_idx+1}/{len(train_loader)}] Loss: {losses.item():.4f}")

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # -------- Save model checkpoint --------
    checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pth"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# ----------- Done -----------
print("Training complete.")
