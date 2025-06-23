import modal
import os
import yaml
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import ToTensor
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import xml.etree.ElementTree as ET
from PIL import Image
from dataclass_implementation import CustomVOCDataset  # <-- your dataset class
import pathlib

app = modal.App("drone-detect-train")

volume = modal.Volume.from_name("my-volume")
VOL_MOUNT_PATH = pathlib.Path("/volume")

custom_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("PyYAML==6.0.2")
    .pip_install("torchvision==0.21.0")
    .pip_install("torch==2.6.0")
    .pip_install("pillow==11.1.0")
    .add_local_file("config.yaml", remote_path="/root/config.yaml")
    .add_local_dir("C:/Users/abhia/Documents/Iot/dataset/DUT-Anti-UAV/detection/train/","/root/dataset/train/")
    .add_local_file("dataclass_implementation.py", remote_path="/root/dataclass_implementation.py")
)

@app.function(image=custom_image,gpu="H100",timeout=4200,volumes={VOL_MOUNT_PATH: volume})
def train_model():

    out_dir = str(VOL_MOUNT_PATH/"model")

    # ----------- Load Config -----------
    with open("/root/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # ----------- Device Setup -----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT

    # ----------- Dataset & DataLoader -----------
    classes = ['__background__', 'UAV']
    train_dataset = CustomVOCDataset(
        images_dir="/root/dataset/train/images",
        annotations_dir="/root/dataset/train/annotations",
        classes=classes,
        transforms=ToTensor()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # <<< ADDED END >>>
    
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
        checkpoint_path = f"{out_dir}/model_epoch_{epoch+1}.pt"
        os.makedirs(out_dir, exist_ok=True)  # Create the right directory
        torch.save(model, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # ----------- Done -----------
    print("Training complete.")


@app.local_entrypoint()
def main():
    train_model.remote()