#!/usr/bin/env python
"""train_weather.py – Weather classifier with class‑weighted CE or Focal Loss,
Balanced sampler, minority augmentations, and W&B sweep‑ready logging
(including **val_macro_f1**).
"""

import argparse, os, time, torch, wandb
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.nn.functional import cross_entropy
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _WeightedLoss
import torch.nn as nn

from dataloader.BDDDataLoader import BDD100KDataset, weather_map
from models.weather_classifier import WeatherClassifier

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

# ---------------------------------------------------------------------------
# Loss definitions
# ---------------------------------------------------------------------------
class FocalLoss(_WeightedLoss):
    def __init__(self, weight=None, gamma: float = 2.0):
        super().__init__(weight=weight)
        self.gamma = gamma
        self.ce = CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# ---------------------------------------------------------------------------
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomRain(p=0.3),
            A.RandomSnow(p=0.3),
            A.RandomFog(p=0.3),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ])

# ---------------------------------------------------------------------------
# Helper: build DataLoader with balanced sampler (for train)
# ---------------------------------------------------------------------------
def make_loader(img_dir, json_path, batch, shuffle, transforms, balanced=False):
    ds = BDD100KDataset(
        image_dir=img_dir,
        label_json_path=json_path,
        transform=transforms,
        filter_weather=None,
    )
    if balanced:
        labels = [weather_map[a["attributes"]["weather"]] for a in ds.annotations if a.get("attributes", {}).get("weather") in weather_map]
        counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=6)
        sample_weights = 1.0 / counts[labels].float()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        return DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=2,
                          collate_fn=lambda x: tuple(zip(*x)), pin_memory=True)
    else:
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=2,
                          collate_fn=lambda x: tuple(zip(*x)), pin_memory=True)

# ---------------------------------------------------------------------------
# Epoch routine
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_tgts = [], []

    for imgs, targets in tqdm(loader, total=len(loader), leave=False):
        imgs = torch.stack(imgs).to(device)
        labels = torch.cat([t["weather"] for t in targets]).to(device)

        preds = model(imgs)
        loss = criterion(preds, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_sz = imgs.size(0)
        loss_sum += loss.item() * batch_sz
        total += batch_sz
        correct += (preds.argmax(1) == labels).sum().item()

        if not is_train:
            all_preds.extend(preds.argmax(1).cpu().tolist())
            all_tgts.extend(labels.cpu().tolist())

    avg_loss = loss_sum / total
    accuracy = correct / total

    if is_train:
        return avg_loss, accuracy, None
    else:
        macro_f1 = f1_score(all_tgts, all_preds, average="macro")
        return avg_loss, accuracy, macro_f1

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(
        args.img_dir, args.label_json, args.batch,
        shuffle=True, transforms=get_transforms(is_train=True),
        balanced=True
    )

    val_loader = make_loader(
        args.val_img_dir, args.val_label_json, args.batch,
        shuffle=False, transforms=get_transforms(is_train=False),
        balanced=False
    )

    model = WeatherClassifier(freeze_backbone=True).to(device)

    weights = torch.load("weather_class_weights.pt").to(device)

    if args.loss_fn == "ce":
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = FocalLoss(weight=weights, gamma=args.focal_gamma)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    wandb.init(project="DriveSafeNet", name=args.run_name, config=vars(args))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc, val_macro_f1 = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()

        wandb.log({"epoch": epoch,
                   "train_loss": train_loss, "train_acc": train_acc,
                   "val_loss": val_loss, "val_acc": val_acc,
                   "val_macro_f1": val_macro_f1})

        print(f"[{epoch}/{args.epochs}] "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
              f"macro_f1={val_macro_f1:.3f} time={time.time()-t0:.1f}s")

        ckpt_path = CKPT_DIR / f"weather_ep{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir",            required=True)
    ap.add_argument("--label_json",         required=True)
    ap.add_argument("--val_img_dir",        required=True)
    ap.add_argument("--val_label_json",     required=True)
    ap.add_argument("--epochs", type=int,   default=10)
    ap.add_argument("--batch",  type=int,   default=64)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--loss_fn", choices=["ce", "focal"], default="ce")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--run_name", default="weather-sweep-run")
    args = ap.parse_args()
    main(args)
