"""
Loss Curve Visualization (Slide 7)
===================================
train.log에서 step-level loss를 파싱하여 학습 곡선을 시각화합니다.
GPU/모델 로딩 불필요.

Usage:
    python experiments/scripts/10_loss_curve.py
"""

import os
import re
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "STIXGeneral"],
    "font.size": 10,
    "mathtext.fontset": "stix",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.facecolor": "#F8F9FA",
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

LOG_PATH = "experiments/results/03_sfa_train/train.log"
OUTPUT_DIR = "experiments/figures/fig_loss_curve"


def parse_train_log(log_path):
    """Parse step-level loss, lr, and time from train.log."""
    pattern = re.compile(
        r"\[Epoch (\d+)\] step (\d+)/(\d+) \| loss: ([\d.]+) \| lr: ([\d.e+-]+)"
    )
    records = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch = int(m.group(1))
                step = int(m.group(2))
                total_steps = int(m.group(3))
                loss = float(m.group(4))
                lr = float(m.group(5))
                # Global step: (epoch-1) * total_steps + step
                global_step = (epoch - 1) * total_steps + step
                records.append({
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "total_steps": total_steps,
                    "loss": loss,
                    "lr": lr,
                })
    return records


def plot_loss_curve(records, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    steps = np.array([r["global_step"] for r in records])
    losses = np.array([r["loss"] for r in records])
    lrs = np.array([r["lr"] for r in records])
    epochs = np.array([r["epoch"] for r in records])
    total_per_epoch = records[0]["total_steps"]

    # --- Main Figure: Loss Curve with LR on secondary axis ---
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Loss curve
    color_loss = "#1A73E8"
    ax1.plot(steps, losses, color=color_loss, linewidth=1.5, alpha=0.9, label="Training Loss")
    ax1.set_xlabel("Training Step (global)")
    ax1.set_ylabel("Loss", color=color_loss)
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.set_ylim(bottom=0)

    # Epoch boundaries
    for ep in range(1, 4):
        boundary = ep * total_per_epoch
        ax1.axvline(x=boundary, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax1.text(boundary, ax1.get_ylim()[1] * 0.95, f"  Epoch {ep} end",
                 fontsize=8, color="gray", va="top")

    # Key milestones
    ax1.annotate(f"Start: {losses[0]:.2f}", xy=(steps[0], losses[0]),
                 fontsize=9, color=color_loss,
                 xytext=(steps[0] + 500, losses[0] - 0.2),
                 arrowprops=dict(arrowstyle="->", color=color_loss, lw=0.8))
    ax1.annotate(f"Final: {losses[-1]:.4f}", xy=(steps[-1], losses[-1]),
                 fontsize=9, color=color_loss,
                 xytext=(steps[-1] - 3000, losses[-1] + 0.5),
                 arrowprops=dict(arrowstyle="->", color=color_loss, lw=0.8))

    # LR on secondary axis
    ax2 = ax1.twinx()
    color_lr = "#EA4335"
    ax2.plot(steps, lrs * 1e5, color=color_lr, linewidth=1.0, alpha=0.6, linestyle="--", label="LR (×1e-5)")
    ax2.set_ylabel("Learning Rate (×1e-5)", color=color_lr)
    ax2.tick_params(axis="y", labelcolor=color_lr)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    ax1.set_title("SFA Fine-tuning Loss Curve (3 Epochs, ChartQA 28K)", fontsize=13)
    ax1.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_curve.pdf"))
    fig.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close(fig)

    # --- Inset: Epoch 2-3 detail ---
    fig2, ax = plt.subplots(figsize=(10, 4))
    mask = steps >= total_per_epoch
    ax.plot(steps[mask], losses[mask], color=color_loss, linewidth=1.5)
    ax.set_xlabel("Training Step (global)")
    ax.set_ylabel("Loss")
    ax.set_title("Epoch 2-3 Detail (Convergence Phase)", fontsize=12)
    ax.axvline(x=2 * total_per_epoch, color="gray", linestyle="--", alpha=0.4)
    ax.text(2 * total_per_epoch, ax.get_ylim()[1] * 0.95, "  Epoch 2→3",
            fontsize=8, color="gray", va="top")
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "loss_curve_detail.pdf"))
    fig2.savefig(os.path.join(output_dir, "loss_curve_detail.png"))
    plt.close(fig2)

    print(f"Loss curve saved to {output_dir}/")
    print(f"  loss_curve.png — Full training curve with LR")
    print(f"  loss_curve_detail.png — Epoch 2-3 convergence detail")
    print(f"\n  Total steps logged: {len(records)}")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Reduction:    {(1 - losses[-1]/losses[0])*100:.1f}%")


if __name__ == "__main__":
    records = parse_train_log(LOG_PATH)
    plot_loss_curve(records, OUTPUT_DIR)
