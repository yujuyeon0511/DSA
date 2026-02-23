"""
Experiment 1: Text Density Estimator Training
==============================================
문서 이미지에서 텍스트 밀집도를 예측하는 lightweight CNN 학습.
Pseudo label을 edge detection + thresholding으로 자동 생성.

논문 Section 3.2 (Adaptive Density-Aware Tokenization)의 핵심 컴포넌트.
논문 Figure 2: Density heatmap 시각화에 사용.

Usage:
    conda activate docmllm
    python experiments/scripts/01_density_estimator.py \
        --mode train \
        --data_dirs /NetDisk/juyeon/train/chartQA/ChartQA\ Dataset/train/png \
                    /NetDisk/juyeon/train/dvqa/images \
        --output_dir experiments/results/01_density \
        --epochs 10

    python experiments/scripts/01_density_estimator.py \
        --mode visualize \
        --checkpoint experiments/results/01_density/best.pth \
        --data_dirs /NetDisk/juyeon/train/chartQA/ChartQA\ Dataset/test/png \
        --output_dir experiments/figures/density_maps
"""

import argparse
import os
import glob
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


# ─── Pseudo Label Generation ───

def generate_density_map(image_np, output_size=28):
    """
    이미지에서 텍스트 밀집도 pseudo label 생성.
    1. Canny edge detection
    2. Adaptive thresholding (text/non-text)
    3. Gaussian blur → continuous density
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Adaptive thresholding for text regions
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Combine edges and threshold
    combined = np.maximum(edges, thresh).astype(np.float32) / 255.0

    # Gaussian blur for smooth density
    density = cv2.GaussianBlur(combined, (15, 15), 5.0)

    # Resize to output_size × output_size
    density = cv2.resize(density, (output_size, output_size), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    if density.max() > 0:
        density = density / density.max()

    return density


# ─── Dataset ───

class DensityDataset(Dataset):
    def __init__(self, image_paths, img_size=448, map_size=28):
        self.image_paths = image_paths
        self.img_size = img_size
        self.map_size = map_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # fallback: blank image
            img = Image.new("RGB", (self.img_size, self.img_size), (255, 255, 255))

        img_np = np.array(img.resize((self.img_size, self.img_size)))
        density = generate_density_map(img_np, self.map_size)

        img_tensor = self.transform(img)
        density_tensor = torch.from_numpy(density).float().unsqueeze(0)  # [1, H, W]

        return img_tensor, density_tensor


# ─── Model ───

class DensityEstimator(nn.Module):
    """
    Lightweight CNN for text density prediction.
    Input: 448×448 RGB image
    Output: 28×28 density map
    ~1.2M parameters
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 448 → 224
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            # 224 → 112
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            # 112 → 56
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            # 56 → 28
            nn.Conv2d(128, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            # 28 → 28 (refine)
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.features(x)


# ─── Training ───

def train(args):
    # Collect images
    all_images = []
    for d in args.data_dirs:
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]:
            all_images.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))
    random.shuffle(all_images)

    if args.max_images:
        all_images = all_images[:args.max_images]

    # Train/val split
    n_val = max(100, int(len(all_images) * 0.05))
    val_images = all_images[:n_val]
    train_images = all_images[n_val:]

    print(f"Train: {len(train_images)}, Val: {len(val_images)}")

    train_ds = DensityDataset(train_images)
    val_ds = DensityDataset(val_images)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DensityEstimator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    os.makedirs(args.output_dir, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.6f}")

        train_loss /= len(train_loader)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, targets).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best.pth"))
            print(f"  Saved best model (val_loss={val_loss:.6f})")

    # Save final
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final.pth"))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")


# ─── Visualization ───

def visualize(args):
    """Density map 시각화 — 논문 Figure 2용"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DensityEstimator().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_images = []
    for d in args.data_dirs:
        for ext in ["*.png", "*.jpg"]:
            all_images.extend(glob.glob(os.path.join(d, ext)))

    random.seed(42)
    selected = random.sample(all_images, min(20, len(all_images)))

    os.makedirs(args.output_dir, exist_ok=True)

    for i, path in enumerate(selected):
        img = Image.open(path).convert("RGB")
        img_resized = img.resize((448, 448))
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            density = model(img_tensor).squeeze().cpu().numpy()

        # Pseudo label for comparison
        pseudo = generate_density_map(np.array(img_resized), 28)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_resized)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(pseudo, cmap="hot", vmin=0, vmax=1)
        axes[1].set_title("Pseudo GT Density")
        axes[1].axis("off")

        axes[2].imshow(density, cmap="hot", vmin=0, vmax=1)
        axes[2].set_title("Predicted Density")
        axes[2].axis("off")

        fig.savefig(os.path.join(args.output_dir, f"density_{i:03d}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(selected)} visualizations to {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "visualize"], required=True)
    parser.add_argument("--data_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", default="experiments/results/01_density")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "visualize":
        assert args.checkpoint, "--checkpoint required for visualize mode"
        visualize(args)


if __name__ == "__main__":
    main()
