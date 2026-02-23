"""
P2-7: Structural Bias Visualization (Figure 7)
================================================
SFA checkpoint에서 학습된 w_row, w_col, w_dist 값을 시각화합니다.
GPU/모델 로딩 불필요 — checkpoint 파일만 로드.

Usage:
    python experiments/scripts/09_structural_bias_viz.py \
        --checkpoint experiments/results/03_sfa_train/best.pth \
        --output_dir experiments/figures/fig7_structural_bias
"""

import argparse
import os
import sys

import numpy as np
import torch

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


def extract_structural_params(checkpoint_path):
    """
    Checkpoint에서 structural bias parameters를 추출.
    Returns dict: {layer_idx: {w_row, w_col, w_dist, block_embed}}
    """
    state = torch.load(checkpoint_path, map_location="cpu")

    layers = {}
    for key, val in state.items():
        if "structural_bias" not in key:
            continue
        # Key format: vision_model.encoder.layers.{i}.attn.structural_bias.{param}
        parts = key.split(".")
        layer_idx = int(parts[parts.index("layers") + 1])

        if layer_idx not in layers:
            layers[layer_idx] = {}

        param_name = parts[-1]
        if param_name == "weight":
            # block_embed.weight
            param_name = "block_embed"
        layers[layer_idx][param_name] = val.float().numpy()

    print(f"Extracted structural bias from {len(layers)} layers")
    return layers


def plot_structural_bias(layers, output_dir):
    """Create Figure 7: Structural bias visualization."""
    os.makedirs(output_dir, exist_ok=True)

    sorted_layers = sorted(layers.keys())
    n_layers = len(sorted_layers)
    num_heads = layers[sorted_layers[0]]["w_row"].shape[0]

    # Collect per-layer, per-head values
    w_rows = np.array([layers[l]["w_row"] for l in sorted_layers])  # [L, H]
    w_cols = np.array([layers[l]["w_col"] for l in sorted_layers])
    w_dists = np.array([layers[l]["w_dist"] for l in sorted_layers])

    # ─── Figure 7a: Heatmap of w_row, w_col, w_dist across layers ───
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

    layer_labels = [str(l) for l in sorted_layers]

    for ax, data, title, cmap in zip(
        axes,
        [w_rows, w_cols, w_dists],
        [r"$w_{row}$", r"$w_{col}$", r"$w_{dist}$"],
        ["RdBu_r", "RdBu_r", "PuOr_r"],
    ):
        vmax = max(abs(data.min()), abs(data.max()))
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_xlabel("Head Index")
        ax.set_title(title, fontsize=14)
        ax.set_xticks(range(0, num_heads, 2))
        ax.set_xticklabels([str(i) for i in range(0, num_heads, 2)])
        plt.colorbar(im, ax=ax, shrink=0.8)

    axes[0].set_ylabel("Layer Index")
    axes[0].set_yticks(range(n_layers))
    axes[0].set_yticklabels(layer_labels)

    fig.suptitle("Figure 7(a): Learned Structural Bias Weights per Layer & Head",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig7a_bias_heatmap.pdf"))
    fig.savefig(os.path.join(output_dir, "fig7a_bias_heatmap.png"))
    plt.close(fig)

    # ─── Figure 7b: Layer-wise mean magnitude ───
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(n_layers)
    width = 0.25

    row_means = np.abs(w_rows).mean(axis=1)
    col_means = np.abs(w_cols).mean(axis=1)
    dist_means = np.abs(w_dists).mean(axis=1)

    ax.bar(x - width, row_means, width, label=r"$|w_{row}|$", color="#1A73E8", alpha=0.85)
    ax.bar(x, col_means, width, label=r"$|w_{col}|$", color="#EA4335", alpha=0.85)
    ax.bar(x + width, dist_means, width, label=r"$|w_{dist}|$", color="#34A853", alpha=0.85)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Absolute Weight")
    ax.set_title("Figure 7(b): Layer-wise Mean |Structural Bias|", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig7b_bias_bar.pdf"))
    fig.savefig(os.path.join(output_dir, "fig7b_bias_bar.png"))
    plt.close(fig)

    # ─── Figure 7c: Example structural bias matrix (last layer, head 0) ───
    last_layer = sorted_layers[-1]
    params = layers[last_layer]
    H_p = W_p = 32
    N = H_p * W_p

    rows = np.arange(H_p).reshape(-1, 1).repeat(W_p, axis=1).reshape(N)
    cols = np.arange(W_p).reshape(1, -1).repeat(H_p, axis=0).reshape(N)

    same_row = (rows[:, None] == rows[None, :]).astype(np.float32)
    same_col = (cols[:, None] == cols[None, :]).astype(np.float32)
    row_dist = np.abs(rows[:, None] - rows[None, :]).astype(np.float32)
    col_dist = np.abs(cols[:, None] - cols[None, :]).astype(np.float32)
    manhattan = (row_dist + col_dist) / (H_p + W_p)

    head = 0
    bias_matrix = (
        params["w_row"][head] * same_row +
        params["w_col"][head] * same_col +
        (-abs(params["w_dist"][head])) * manhattan
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    # Show center 128x128 patch for visibility
    center = N // 2
    window = 64
    sub = bias_matrix[center-window:center+window, center-window:center+window]
    vmax = max(abs(sub.min()), abs(sub.max()))
    im = ax.imshow(sub, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Figure 7(c): Structural Bias Matrix\nLayer {last_layer}, Head {head} (center 128×128)",
                 fontsize=12)
    ax.set_xlabel("Patch Index")
    ax.set_ylabel("Patch Index")
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig7c_bias_matrix.pdf"))
    fig.savefig(os.path.join(output_dir, "fig7c_bias_matrix.png"))
    plt.close(fig)

    print(f"\nFigure 7 saved to {output_dir}/")
    print(f"  fig7a_bias_heatmap.png — Weight heatmap across layers/heads")
    print(f"  fig7b_bias_bar.png — Layer-wise mean magnitude")
    print(f"  fig7c_bias_matrix.png — Example bias matrix")

    # Print summary statistics
    print(f"\n  Structural Bias Summary:")
    print(f"  {'Layer':<8} {'|w_row|':>10} {'|w_col|':>10} {'|w_dist|':>10}")
    print(f"  {'-'*40}")
    for i, l in enumerate(sorted_layers):
        print(f"  {l:<8} {row_means[i]:>10.4f} {col_means[i]:>10.4f} {dist_means[i]:>10.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="experiments/results/03_sfa_train/best.pth")
    parser.add_argument("--output_dir",
                        default="experiments/figures/fig7_structural_bias")
    args = parser.parse_args()

    layers = extract_structural_params(args.checkpoint)
    plot_structural_bias(layers, args.output_dir)


if __name__ == "__main__":
    main()
