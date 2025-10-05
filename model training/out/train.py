# train.py
import os, glob, math, time, argparse
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from utils import (ensure_dir, load_gray_png, save_png, points_csv_to_mask,
                   make_training_tiles, normalize)

# ---------- Small U-Net ----------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,1,1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,1,1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.net(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=1, ch=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, ch)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(ch, ch*2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(ch*2, ch*4)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(ch*4, ch*8)

        self.u3 = nn.ConvTranspose2d(ch*8, ch*4, 2,2)
        self.c3 = DoubleConv(ch*8, ch*4)
        self.u2 = nn.ConvTranspose2d(ch*4, ch*2, 2,2)
        self.c2 = DoubleConv(ch*4, ch*2)
        self.u1 = nn.ConvTranspose2d(ch*2, ch, 2,2)
        self.c1 = DoubleConv(ch*2, ch)
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self,x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        d4 = self.d4(self.p3(d3))
        x = self.u3(d4)
        x = self.c3(torch.cat([x,d3], dim=1))
        x = self.u2(x)
        x = self.c2(torch.cat([x,d2], dim=1))
        x = self.u1(x)
        x = self.c1(torch.cat([x,d1], dim=1))
        return self.out(x)

# ---------- Losses ----------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps=eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2*(probs*targets).sum(dim=(2,3)) + self.eps
        den = (probs+targets).sum(dim=(2,3)) + self.eps
        dice = 1 - num/den
        return dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha=alpha; self.gamma=gamma
    def forward(self, logits, targets):
        # targets 0/1
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = targets*p + (1-targets)*(1-p)
        w = self.alpha*targets + (1-self.alpha)*(1-targets)
        loss = w * (1-pt).pow(self.gamma) * bce
        return loss.mean()

# ---------- Dataset ----------
class TileDataset(Dataset):
    def __init__(self, image_paths, mask_paths, tile=256, stride=256,
                 augment=True, pos_frac=0.7, max_tiles_per_img=3000, from_points=False):
        self.samples=[]
        self.tile=tile
        self.aug = A.Compose([
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
            A.GaussNoise(var_limit=(5.0,30.0), p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, border_mode=0, p=0.5),
            ToTensorV2(),
        ]) if augment else A.Compose([ToTensorV2()])

        for img_path in image_paths:
            base = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = mask_paths.get(base, None)
            img = load_gray_png(img_path)
            H,W = img.shape

            if mask_path and os.path.exists(mask_path):
                mask = (load_gray_png(mask_path)>0).astype(np.uint8)
            else:
                csv = f"data/points/{base}.csv"
                mask = points_csv_to_mask(csv, H, W, radius=3)

            tiles = make_training_tiles(img, mask, tile=tile, stride=stride,
                                        pos_frac=pos_frac, max_tiles=max_tiles_per_img, pos_thresh=20)
            for (x,y) in tiles:
                self.samples.append((img_path, mask_path, x,y))

        self.mask_lookup = mask_paths

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, x, y = self.samples[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        img = load_gray_png(img_path)
        H,W = img.shape
        if mask_path and os.path.exists(mask_path):
            mask = (load_gray_png(mask_path)>0).astype(np.uint8)
        else:
            mask = points_csv_to_mask(f"data/points/{base}.csv", H, W, radius=3)

        patch = img[y:y+self.tile, x:x+self.tile]
        mpatch = mask[y:y+self.tile, x:x+self.tile]
        patch = normalize(patch)
        patch = (patch*255).astype(np.uint8)

        augmented = self.aug(image=patch, mask=mpatch)
        ip = augmented['image'].float()/255.0
        mp = augmented['mask'].float().unsqueeze(0)  # to 1xHxW
        return ip.unsqueeze(0), mp  # [1,H,W], [1,H,W]

def split_paths(all_imgs, val_ratio=0.1):
    all_imgs = sorted(all_imgs)
    n = len(all_imgs)
    v = max(1, int(n*val_ratio))
    return all_imgs[v:], all_imgs[:v]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default="data/images")
    ap.add_argument("--mask_dir", default="data/masks")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", default="models")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    image_paths = sorted(glob.glob(os.path.join(args.img_dir, "*.png")))
    mask_paths = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(args.mask_dir, "*.png"))}

    train_imgs, val_imgs = split_paths(image_paths, val_ratio=0.15)
    train_ds = TileDataset(train_imgs, mask_paths, tile=args.tile, stride=args.stride, augment=True)
    val_ds   = TileDataset(val_imgs,   mask_paths, tile=args.tile, stride=args.stride, augment=False, max_tiles_per_img=1000, pos_frac=0.5)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetSmall(in_ch=1, ch=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    dice = DiceLoss()
    focal = FocalLoss(alpha=0.25, gamma=2.0)

    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    best_val = float("inf")
    for ep in range(1, args.epochs+1):
        model.train()
        tr_loss=0.0
        for x,y in tqdm(train_dl, desc=f"Epoch {ep}/{args.epochs} [train]"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = model(x)
                loss = 0.5*dice(logits, y) + 0.5*focal(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item()*x.size(0)
        tr_loss/=len(train_dl.dataset)

        model.eval()
        va_loss=0.0
        with torch.no_grad():
            for x,y in tqdm(val_dl, desc=f"Epoch {ep}/{args.epochs} [val]"):
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = 0.5*dice(logits, y) + 0.5*focal(logits, y)
                va_loss += loss.item()*x.size(0)
        va_loss/=max(1,len(val_dl.dataset))
        sched.step()

        print(f"Epoch {ep}: train {tr_loss:.4f}  val {va_loss:.4f}")
        torch.save({"model": model.state_dict()}, os.path.join(args.out_dir, "last.pt"))
        if va_loss<best_val:
            best_val=va_loss
            torch.save({"model": model.state_dict()}, os.path.join(args.out_dir, "best.pt"))
            print("  ✓ saved best")

if __name__=="__main__":
    main()
