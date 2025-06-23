import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class CustomVOCDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, classes, transforms=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
        self.classes = classes  # List like ['__background__', 'person', 'car', ...]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        image = Image.open(img_path).convert("RGB")

        # Parse corresponding XML annotation
        annotation_path = os.path.join(self.annotations_dir, img_filename.replace(".jpg", ".xml"))
        boxes = []
        labels = []

        tree = ET.parse(annotation_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in self.classes:
                continue  # skip unknown labels

            label_idx = self.classes.index(label)

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_idx)

        # After parsing boxes and labels
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # Required by Faster R-CNN
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        # Optional: add area and iscrowd if needed
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transforms:
            image = self.transforms(image)

        return image, target
