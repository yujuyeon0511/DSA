"""
Figure 1: Motivation — Uniform vs. Adaptive Patching
=====================================================
Generates a 1x3 panel figure for the paper's motivation section.

(a) Original image with a uniform 32x32 patch grid
(b) Information density heatmap overlay (inferno colormap)
(c) Adaptive patching: dense->8x8, medium->14x14, sparse->28x28

Usage:
    python experiments/scripts/05_figure_motivation.py \
        --image experiments/figures/sample_images/chartqa_sample.png \
        --output_dir experiments/figures/motivation

    # With pre-trained density estimator checkpoint:
    python experiments/scripts/05_figure_motivation.py \
        --image experiments/figures/sample_images/chartqa_sample.png \
        --density_ckpt experiments/results/01_density/best.pth \
        --output_dir experiments/figures/motivation
"""

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "STIXGeneral"],
    "font.size": 10,
    "mathtext.fontset": "stix",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "#F8F9FA",
})

import argparse
import os
import sys
import numpy as np
import cv2

import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ---------------------------------------------------------------------------
# Density estimation: use CNN checkpoint if provided, otherwise edge-based
# ---------------------------------------------------------------------------

def _load_density_estimator(ckpt_path):
    """Load the DensityEstimator CNN from 01_density_estimator.py."""
    import torch
    # Import the sibling module
    sys.path.insert(0, os.path.dirname(__file__))
    _de = __import__("01_density_estimator")

    model = _de.DensityEstimator()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def compute_density_map(image_np, ckpt_path=None, grid_size=32):
    """
    Return a *grid_size x grid_size* density map in [0, 1].

    If *ckpt_path* is given, run the CNN model.
    Otherwise, fall back to the pseudo-label generator (edge detection).
    """
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        import torch
        from torchvision import transforms

        model = _load_density_estimator(ckpt_path)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        tensor = transform(image_np).unsqueeze(0)
        with torch.no_grad():
            density = model(tensor).squeeze().cpu().numpy()
        # The CNN outputs 28x28; resize to grid_size
        density = cv2.resize(density, (grid_size, grid_size),
                             interpolation=cv2.INTER_LINEAR)
    else:
        # Pseudo-label from 01_density_estimator.generate_density_map
        sys.path.insert(0, os.path.dirname(__file__))
        _de = __import__("01_density_estimator")
        density = _de.generate_density_map(image_np, output_size=grid_size)

    # Ensure [0, 1]
    dmin, dmax = density.min(), density.max()
    if dmax - dmin > 1e-8:
        density = (density - dmin) / (dmax - dmin)
    else:
        density = np.zeros_like(density)
    return density


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

IMG_SIZE = 448          # Canonical image dimension for the figure
GRID_N = 32             # Number of uniform patches per axis
PATCH_PX = IMG_SIZE // GRID_N  # = 14 pixels per patch


def _draw_grid(ax, nx, ny, color="gray", linewidth=0.5, alpha=0.6):
    """Draw a regular grid on *ax* covering the [0, IMG_SIZE] extent."""
    step_x = IMG_SIZE / nx
    step_y = IMG_SIZE / ny
    for i in range(nx + 1):
        ax.axvline(i * step_x, color=color, linewidth=linewidth, alpha=alpha)
    for j in range(ny + 1):
        ax.axhline(j * step_y, color=color, linewidth=linewidth, alpha=alpha)


def panel_a(ax, image):
    """(a) Original image with uniform 32x32 grid."""
    ax.imshow(image)
    _draw_grid(ax, GRID_N, GRID_N, color="gray", linewidth=0.4, alpha=0.55)
    ax.set_title("(a) Original + Uniform Grid")
    ax.set_xlim(0, IMG_SIZE)
    ax.set_ylim(IMG_SIZE, 0)
    ax.set_xticks([])
    ax.set_yticks([])


def panel_b(ax, image, density):
    """(b) Image overlaid with per-patch density (inferno colormap)."""
    ax.imshow(image)

    cmap = plt.cm.inferno
    norm = Normalize(vmin=0, vmax=1)

    rows, cols = density.shape
    step_y = IMG_SIZE / rows
    step_x = IMG_SIZE / cols

    for r in range(rows):
        for c in range(cols):
            val = density[r, c]
            color = cmap(norm(val))
            rect = mpatches.Rectangle(
                (c * step_x, r * step_y), step_x, step_y,
                linewidth=0, edgecolor="none",
                facecolor=color, alpha=0.4,
            )
            ax.add_patch(rect)

    ax.set_title("(b) Information Density")
    ax.set_xlim(0, IMG_SIZE)
    ax.set_ylim(IMG_SIZE, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Density", fontsize=10)
    cbar.ax.tick_params(labelsize=8)


def panel_c(ax, image, density):
    """
    (c) Adaptive patching.

    Density thresholds (per 14x14 patch):
        dense   (>0.5)  : subdivide into 4x4 sub-patches  -> effective 8x8 px
        medium  (0.2-0.5): keep 14x14 patch
        sparse  (<0.2)  : merge 2x2 patches               -> effective 28x28 px

    The density map is grid_size x grid_size (=32).  We process it in the
    canonical 32-patch grid and draw rectangles accordingly.
    """
    ax.imshow(image)

    rows, cols = density.shape  # 32 x 32
    step = IMG_SIZE / rows      # 14 px

    # Track which cells have already been drawn (for sparse merging)
    drawn = np.zeros((rows, cols), dtype=bool)

    # --- Pass 1: sparse merge (< 0.2) — try to merge 2x2 blocks ----------
    for r in range(0, rows - 1, 2):
        for c in range(0, cols - 1, 2):
            block = density[r:r+2, c:c+2]
            if np.all(block < 0.2):
                rect = mpatches.Rectangle(
                    (c * step, r * step), 2 * step, 2 * step,
                    linewidth=0.8, edgecolor="#2196F3",
                    facecolor="#2196F3", alpha=0.12,
                )
                ax.add_patch(rect)
                # Border
                rect_b = mpatches.Rectangle(
                    (c * step, r * step), 2 * step, 2 * step,
                    linewidth=0.8, edgecolor="#2196F3",
                    facecolor="none",
                )
                ax.add_patch(rect_b)
                drawn[r:r+2, c:c+2] = True

    # --- Pass 2: dense (> 0.5) and medium (0.2–0.5) cells -----------------
    for r in range(rows):
        for c in range(cols):
            if drawn[r, c]:
                continue
            val = density[r, c]
            x0 = c * step
            y0 = r * step

            if val > 0.5:
                # Dense: draw 4x4 sub-grid inside this patch (-> ~3.5 px cells)
                sub_n = 4
                sub_step = step / sub_n
                for sr in range(sub_n):
                    for sc in range(sub_n):
                        rect = mpatches.Rectangle(
                            (x0 + sc * sub_step, y0 + sr * sub_step),
                            sub_step, sub_step,
                            linewidth=0.4, edgecolor="#E53935",
                            facecolor="#E53935", alpha=0.10,
                        )
                        ax.add_patch(rect)
                # Outer border for the patch
                rect_b = mpatches.Rectangle(
                    (x0, y0), step, step,
                    linewidth=0.6, edgecolor="#E53935",
                    facecolor="none",
                )
                ax.add_patch(rect_b)
                drawn[r, c] = True
            else:
                # Medium: plain 14x14 patch
                rect = mpatches.Rectangle(
                    (x0, y0), step, step,
                    linewidth=0.5, edgecolor="#43A047",
                    facecolor="#43A047", alpha=0.08,
                )
                ax.add_patch(rect)
                drawn[r, c] = True

    ax.set_title("(c) Adaptive Patching (Ours)")
    ax.set_xlim(0, IMG_SIZE)
    ax.set_ylim(IMG_SIZE, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor="#E53935", alpha=0.35,
                       edgecolor="#E53935", label="Dense (8\u00d78)"),
        mpatches.Patch(facecolor="#43A047", alpha=0.25,
                       edgecolor="#43A047", label="Medium (14\u00d714)"),
        mpatches.Patch(facecolor="#2196F3", alpha=0.25,
                       edgecolor="#2196F3", label="Sparse (28\u00d728)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              frameon=True, framealpha=0.85, edgecolor="#cccccc")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 1 (Motivation) — Uniform vs. Adaptive Patching")
    parser.add_argument(
        "--image",
        default="experiments/figures/sample_images/chartqa_sample.png",
        help="Path to sample chart image",
    )
    parser.add_argument(
        "--density_ckpt",
        default=None,
        help="Path to density estimator checkpoint (optional; uses edge "
             "detection pseudo-label if not provided)",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/figures/motivation",
        help="Directory to save output figure",
    )
    args = parser.parse_args()

    # ---- Load & resize image ------------------------------------------------
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {args.image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_AREA)

    # ---- Density map --------------------------------------------------------
    density = compute_density_map(img_resized, ckpt_path=args.density_ckpt,
                                  grid_size=GRID_N)
    print(f"Density map shape: {density.shape}  "
          f"min={density.min():.3f}  max={density.max():.3f}  "
          f"mean={density.mean():.3f}")

    # ---- Create 1x3 figure --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.15)

    panel_a(axes[0], img_resized)
    panel_b(axes[1], img_resized, density)
    panel_c(axes[2], img_resized, density)

    # ---- Save ---------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    pdf_path = os.path.join(args.output_dir, "figure1_motivation.pdf")
    png_path = os.path.join(args.output_dir, "figure1_motivation.png")

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
