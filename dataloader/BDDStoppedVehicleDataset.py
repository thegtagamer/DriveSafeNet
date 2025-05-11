import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
# Map categories to label IDs (start from 1 because 0 is background)
STOPPED_VEHICLE_MAP = {
    'car': 1,
    'bus': 2,
    'truck': 3
}

# Albumentations transform
def get_stop_vehicle_transform():
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

class BDDStoppedVehicleDataset(Dataset):
    def __init__(self, img_dir, label_json, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms or get_stop_vehicle_transform()

        with open(label_json, 'r') as f:
            all_data = json.load(f)

        self.samples = []
        for ann in all_data:
            boxes = []
            labels = []

            for obj in ann.get("labels", []):
                category = obj.get("category")
                attr = obj.get("attributes", {})
                if category in STOPPED_VEHICLE_MAP:
                    if "box2d" in obj:
                        box = obj["box2d"]
                        boxes.append([box["x1"], box["y1"], box["x2"], box["y2"]])
                        labels.append(STOPPED_VEHICLE_MAP[category])

            if boxes:
                self.samples.append({
                    "image_name": ann["name"],
                    "boxes": boxes,
                    "labels": labels
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.img_dir, sample["image_name"])
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        boxes = sample["boxes"]
        labels = sample["labels"]

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed["image"]
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["labels"], dtype=torch.int64)
        else:
            image = ToTensorV2()(image=image)["image"]
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return image, target

