# BDDDataLoader.py
# -------------------------------------------------------------
# Custom PyTorch Dataset for BDD100K 10 K (or full 100 K) images
# – Skips any annotation whose JPEG is missing
# – Optional recursive search for nested image folders
# – Works with Albumentations pipelines that finish with ToTensorV2
# -------------------------------------------------------------
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

# ---------- label maps ----------
hazard_label_map = {
    'car': 0, 'bus': 1, 'truck': 2, 'train': 3, 'bike': 4,
    'motor': 5, 'rider': 6, 'person': 7, 'traffic light': 8,
    'traffic sign': 9
}

weather_map = {
    'clear': 0, 'partly cloudy': 1, 'overcast': 2,
    'rainy': 3, 'snowy': 4, 'foggy': 5
}

# ---------- dataset ----------
class BDD100KDataset(Dataset):
    """
    Args
    ----
    image_dir      : root folder that contains JPEG images
    label_json_path: path to bdd100k_labels_images_*.json
    transform      : Albumentations or torchvision transforms (should end with ToTensorV2)
    filter_weather : optional set, e.g. {'foggy', 'snowy'}
    recursive      : True → search sub‑directories with rglob
    """
    def __init__(self,
                 image_dir: str,
                 label_json_path: str,
                 transform=None,
                 filter_weather=None,
                 recursive: bool = False):
        self.image_dir = image_dir
        self.transform = transform
        self.recursive = recursive

        # ---- load annotation JSON ----
        with open(label_json_path, "r") as f:
            anns = json.load(f)

        # optional weather filtering
        if filter_weather:
            anns = [a for a in anns
                    if a.get("attributes", {}).get("weather") in filter_weather]

        # ---- build set of existing JPEG file names ----
        if recursive:
            jpg_set = {p.name for p in Path(image_dir).rglob("*.jpg")}
        else:
            jpg_set = set(os.listdir(image_dir))

        # keep only annotations whose image exists
        self.annotations = [a for a in anns if a["name"] in jpg_set]
        dropped = len(anns) - len(self.annotations)
        if dropped:
            print(f"[BDD] Skipped {dropped:,} missing images "
                  f"({len(self.annotations):,} usable).")

    # -------------------------------------------------
    def __len__(self):
        return len(self.annotations)

    # -------------------------------------------------
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_name = ann["name"]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # ---- build target dict ----
        boxes, labels = [], []
        for obj in ann.get("labels", []):
            cat = obj["category"]
            if cat in hazard_label_map and "box2d" in obj:
                b = obj["box2d"]
                boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
                labels.append(hazard_label_map[cat])

        target = {
            "boxes":  torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "weather": torch.tensor([weather_map.get(
                       ann.get("attributes", {}).get("weather", "clear"), 0)])
        }

        # ---- apply transforms ----
        if self.transform:
            out   = self.transform(image=np.array(image))
            image = out["image"]              # tensor from ToTensorV2
        else:
            # ensure tensor even without augmentations
            image = ToTensorV2()(image=np.array(image))["image"]

        return image, target
