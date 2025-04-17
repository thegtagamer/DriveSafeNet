#!/usr/bin/env python
"""
Train a 6‑class weather classifier on BDD100K‑10k.
Usage (from repo root):
python train_weather.py \
  --img_dir images/10k/train \
  --label_json labels/bdd100k_labels_images_train.json \
  --val_img_dir images/10k/val \
  --val_label_json labels/bdd100k_labels_images_val.json
"""
import argparse, os, time, torch, wandb
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataloader.BDDDataLoader import BDD100KDataset, weather_map
from models.weather_classifier import WeatherClassifier

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from pathlib import Path
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

def get_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])

def make_loader(img_dir, json_path, batch, shuffle):
    ds = BDD100KDataset(
        image_dir     = img_dir,
        label_json_path = json_path,
        transform     = get_transforms(),
        filter_weather=None  # use all
    )
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=2,
                      collate_fn=lambda x: tuple(zip(*x)))

def run_epoch(model, loader, optim=None):
    """
    If optim is None → validation mode.
    Returns (avg_loss, top1_accuracy)
    """
    device = next(model.parameters()).device
    is_train = optim is not None
    model.train(is_train)
    total, correct, loss_sum = 0, 0, 0.0
    for imgs, targets in tqdm(loader, total=len(loader), leave=False):
        imgs = torch.stack(imgs).to(device)            # [B,3,224,224]
        labels = torch.cat([t["weather"] for t in targets]).to(device)  # [B]
        
        preds = model(imgs)
        loss  = F.cross_entropy(preds, labels)
        
        if is_train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        loss_sum += loss.item() * imgs.size(0)
        total += imgs.size(0)
        correct += (preds.argmax(1) == labels).sum().item()
    
    return loss_sum / total, correct / total

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = WeatherClassifier(freeze_backbone=True).to(device)
    
    train_loader = make_loader(args.img_dir, args.label_json, args.batch, shuffle=True)
    val_loader   = make_loader(args.val_img_dir, args.val_label_json, args.batch, shuffle=False)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    wandb.init(project="DriveSafeNet", name="weather-v1", config=vars(args))
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, optim=optimizer)
        val_loss, val_acc     = run_epoch(model, val_loader, optim=None)
        scheduler.step()
        
        wandb.log({"epoch": epoch,
                   "train_loss": train_loss, "train_acc": train_acc,
                   "val_loss": val_loss,   "val_acc": val_acc})
        print(f"[{epoch}/{args.epochs}] "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
              f"time={time.time()-t0:.1f}s")
        
        # checkpoint every epoch
        ckpt_path = CKPT_DIR / f"weather_ep{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir",           required=True)
    ap.add_argument("--label_json",        required=True)
    ap.add_argument("--val_img_dir",       required=True)
    ap.add_argument("--val_label_json",    required=True)
    ap.add_argument("--epochs", type=int,  default=3)
    ap.add_argument("--batch",  type=int,  default=64)
    ap.add_argument("--lr",     type=float, default=2e-4)
    args = ap.parse_args()
    main(args)
