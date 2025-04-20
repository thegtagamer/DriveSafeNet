#!/usr/bin/env python3
"""Evaluate weather‑classification checkpoint on BDD100K validation split.

Outputs:
  • console print‑out of classification report (precision / recall / F1 / support)
  • confusion_matrix.png saved in --out_dir
  • metrics.json (overall accuracy and per‑class numbers) in --out_dir

Usage (example):

  python evaluate_weather.py \
      --img_dir data/bdd100k/images/100k/val \
      --label_json data/bdd100k/labels/det_20/det_val.json \
      --ckpt checkpoints/weather_ep3.pth \
      --batch 64 --num_workers 4 \
      --out_dir eval_results
"""
import argparse, json, os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- project imports ----
from dataloader.BDDDataLoader import BDD100KDataset, weather_map
from models.weather_classifier import WeatherClassifier  # adjust if your class name differs
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2


# -------------------------------------------------------------
# helpers
# -------------------------------------------------------------

def get_val_transforms():
    return Compose([
        Resize(224, 224),  # keep in sync with training
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def load_model(ckpt_path: str, num_classes: int):
    model = WeatherClassifier(num_classes=num_classes)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# -------------------------------------------------------------
# main
# -------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", required=True)
parser.add_argument("--label_json", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--recursive", action="store_true", help="set if images in nested folders")
parser.add_argument("--out_dir", default="eval_results")
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# dataset & loader
val_ds = BDD100KDataset(
    image_dir=args.img_dir,
    label_json_path=args.label_json,
    transform=get_val_transforms(),
    recursive=args.recursive,
)
val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=lambda x: tuple(zip(*x)))

print(f"Validation images: {len(val_ds):,}")

# model
classes = list(weather_map.keys())  # ['clear', 'partly cloudy', ...]
model = load_model(args.ckpt, num_classes=len(classes)).to(args.device)

# inference loop
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, targets in tqdm(val_loader):
        imgs = torch.stack(imgs).to(args.device)
        labels = torch.tensor([t["weather"].item() for t in targets]).to(args.device)

        logits = model(imgs)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

# metrics
report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
print("\n" + report)

# overall accuracy
acc = (all_preds == all_labels).mean()
print(f"Overall accuracy: {acc:.4f}")

# save metrics json
(metrics_path := out_dir / "metrics.json").write_text(json.dumps({
    "accuracy": float(acc),
    "classification_report": report,
}))
print(f"Saved metrics to {metrics_path}")

# confusion matrix
cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticklabels(classes)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() * 0.5 else "black")

ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("BDD100K Weather – Confusion Matrix")
fig.colorbar(im, fraction=0.046, pad=0.04)
cm_path = out_dir / "confusion_matrix.png"
fig.tight_layout()
fig.savefig(cm_path, dpi=300)
print(f"Saved confusion matrix to {cm_path}")

