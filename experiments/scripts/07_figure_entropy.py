"""
Figure 5: Attention Entropy Analysis — Visualization Script
============================================================
Pre-computed entropy JSON을 읽어 publication-quality figure를 생성합니다.

Panel (a): Region별 entropy 분포 (violin/box plot)
Panel (b): Layer-wise entropy 변화 (line plot)

Usage:
    conda activate docmllm

    # Baseline only
    python experiments/scripts/07_figure_entropy.py \
        --baseline_data experiments/results/04_analysis/entropy_analysis.json

    # Baseline vs SFA comparison
    python experiments/scripts/07_figure_entropy.py \
        --baseline_data experiments/results/04_analysis/entropy_analysis.json \
        --sfa_data experiments/results/04_analysis/entropy_analysis_sfa.json \
        --output_dir experiments/figures/fig5_entropy
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "STIXGeneral"],
    "font.size": 10,
    "mathtext.fontset": "stix",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.facecolor": "#F8F9FA",
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Try to import seaborn; fall back to matplotlib box plots if unavailable
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Color palette
COLOR_SLATE = "#70757A"
COLOR_BLUE = "#1A73E8"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_entropy_data(path):
    """Load entropy analysis JSON and return the list of records."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def extract_region_entropies(records):
    """Return (text_entropies, sparse_entropies) lists from all records."""
    text = [r["text_entropy"] for r in records]
    sparse = [r["sparse_entropy"] for r in records]
    return text, sparse


def extract_layerwise(records):
    """Return {layer: (mean_text, std_text, mean_sparse, std_sparse)}."""
    by_layer = defaultdict(lambda: {"text": [], "sparse": []})
    for r in records:
        by_layer[r["layer"]]["text"].append(r["text_entropy"])
        by_layer[r["layer"]]["sparse"].append(r["sparse_entropy"])

    result = {}
    for layer in sorted(by_layer.keys()):
        t = np.array(by_layer[layer]["text"])
        s = np.array(by_layer[layer]["sparse"])
        result[layer] = {
            "text_mean": float(np.mean(t)),
            "text_std": float(np.std(t)),
            "sparse_mean": float(np.mean(s)),
            "sparse_std": float(np.std(s)),
        }
    return result


# ---------------------------------------------------------------------------
# Annotation helper
# ---------------------------------------------------------------------------

def annotate_mean(ax, data, x_pos, color="black", fontsize=8):
    """Place a small mean-value annotation above the distribution."""
    mean_val = np.mean(data)
    ax.text(
        x_pos, mean_val, f"  {mean_val:.2f}",
        ha="left", va="center", fontsize=fontsize, color=color,
        fontweight="bold",
    )


# ---------------------------------------------------------------------------
# Panel (a): Entropy Distribution by Region
# ---------------------------------------------------------------------------

def plot_panel_a(ax, baseline_records, sfa_records=None):
    """Violin / box plot of text-dense vs sparse entropy."""
    b_text, b_sparse = extract_region_entropies(baseline_records)

    if sfa_records is not None:
        s_text, s_sparse = extract_region_entropies(sfa_records)
        labels = ["Baseline\nText", "Baseline\nSparse",
                   "SFA\nText", "SFA\nSparse"]
        all_data = [b_text, b_sparse, s_text, s_sparse]
        colors = [COLOR_SLATE, COLOR_SLATE, COLOR_BLUE, COLOR_BLUE]
        hatches = [None, "///", None, "///"]
    else:
        labels = ["Text-dense", "Sparse"]
        all_data = [b_text, b_sparse]
        colors = [COLOR_SLATE, COLOR_SLATE]
        hatches = [None, "///"]

    positions = list(range(1, len(all_data) + 1))

    if HAS_SEABORN:
        # Build a long-form dataframe-like structure for seaborn
        import pandas as pd

        rows = []
        for vals, label in zip(all_data, labels):
            for v in vals:
                rows.append({"Entropy": v, "Region": label})
        df = pd.DataFrame(rows)

        palette = {lab: col for lab, col in zip(labels, colors)}
        parts = sns.violinplot(
            x="Region", y="Entropy", data=df, ax=ax,
            palette=palette, inner="box", linewidth=1.0,
            cut=0, scale="width",
        )
        # Annotate means
        for i, (vals, label) in enumerate(zip(all_data, labels)):
            mean_val = np.mean(vals)
            ax.text(
                i, mean_val, f" {mean_val:.2f}",
                ha="left", va="center", fontsize=8,
                color="black", fontweight="bold",
            )
    else:
        bp = ax.boxplot(
            all_data, positions=positions, widths=0.5,
            patch_artist=True, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=5),
            medianprops=dict(color="white", linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
        )
        for patch, color, hatch in zip(bp["boxes"], colors, hatches):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            if hatch:
                patch.set_hatch(hatch)
                patch.set_edgecolor(color)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

        # Annotate means
        for pos, vals in zip(positions, all_data):
            mean_val = np.mean(vals)
            ax.text(
                pos + 0.3, mean_val, f"{mean_val:.2f}",
                ha="left", va="center", fontsize=8,
                color="black", fontweight="bold",
            )

    ax.set_ylabel(r"Attention Entropy $H(\alpha)$")
    ax.set_title("(a)  Entropy Distribution by Region")
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Panel (b): Layer-wise Entropy
# ---------------------------------------------------------------------------

def plot_panel_b(ax, baseline_records, sfa_records=None):
    """Line plot of entropy across layers (last 4)."""
    b_lw = extract_layerwise(baseline_records)
    layers = sorted(b_lw.keys())
    x = np.array(layers)

    b_text_mean = np.array([b_lw[l]["text_mean"] for l in layers])
    b_text_std = np.array([b_lw[l]["text_std"] for l in layers])
    b_sparse_mean = np.array([b_lw[l]["sparse_mean"] for l in layers])
    b_sparse_std = np.array([b_lw[l]["sparse_std"] for l in layers])

    # Baseline lines
    ax.errorbar(
        x, b_text_mean, yerr=b_text_std,
        fmt="o--", color=COLOR_SLATE, linewidth=1.8, markersize=6,
        capsize=3, label="Baseline — Text",
    )
    ax.errorbar(
        x, b_sparse_mean, yerr=b_sparse_std,
        fmt="s-", color=COLOR_SLATE, linewidth=1.8, markersize=6,
        capsize=3, label="Baseline — Sparse",
    )

    if sfa_records is not None:
        s_lw = extract_layerwise(sfa_records)
        s_layers = sorted(s_lw.keys())
        sx = np.array(s_layers)

        s_text_mean = np.array([s_lw[l]["text_mean"] for l in s_layers])
        s_text_std = np.array([s_lw[l]["text_std"] for l in s_layers])
        s_sparse_mean = np.array([s_lw[l]["sparse_mean"] for l in s_layers])
        s_sparse_std = np.array([s_lw[l]["sparse_std"] for l in s_layers])

        ax.errorbar(
            sx, s_text_mean, yerr=s_text_std,
            fmt="o--", color=COLOR_BLUE, linewidth=1.8, markersize=6,
            capsize=3, label="SFA — Text",
        )
        ax.errorbar(
            sx, s_sparse_mean, yerr=s_sparse_std,
            fmt="s-", color=COLOR_BLUE, linewidth=1.8, markersize=6,
            capsize=3, label="SFA — Sparse",
        )

    ax.set_xlabel("Layer (last 4)")
    ax.set_ylabel("Entropy")
    ax.set_title("(b)  Layer-wise Entropy")
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="both", alpha=0.3)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(name, records):
    """Print per-layer and overall summary statistics to stdout."""
    text_all, sparse_all = extract_region_entropies(records)
    lw = extract_layerwise(records)
    layers = sorted(lw.keys())

    n_images = len(set(r["image"] for r in records))
    n_records = len(records)
    avg_text_ratio = np.mean([r["text_ratio"] for r in records])

    print(f"\n{'=' * 60}")
    print(f"  {name}  ({n_images} images, {n_records} measurements)")
    print(f"{'=' * 60}")
    print(f"  Avg text-region ratio: {avg_text_ratio:.3f}")
    print(f"  Overall text entropy:  {np.mean(text_all):.4f}  (std {np.std(text_all):.4f})")
    print(f"  Overall sparse entropy: {np.mean(sparse_all):.4f}  (std {np.std(sparse_all):.4f})")
    delta = np.mean(text_all) - np.mean(sparse_all)
    print(f"  Delta (text - sparse):  {delta:+.4f}")

    print(f"\n  {'Layer':<8} {'Text mean':>11} {'Sparse mean':>13} {'Delta':>9}")
    print(f"  {'-'*44}")
    for l in layers:
        d = lw[l]
        dd = d["text_mean"] - d["sparse_mean"]
        print(f"  {l:<8} {d['text_mean']:>11.4f} {d['sparse_mean']:>13.4f} {dd:>+9.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Figure 5 — Attention Entropy Analysis visualization",
    )
    parser.add_argument(
        "--baseline_data", type=str, required=True,
        help="Path to baseline entropy JSON (e.g. entropy_analysis.json)",
    )
    parser.add_argument(
        "--sfa_data", type=str, default=None,
        help="Path to SFA entropy JSON for comparison (optional)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiments/figures/fig5_entropy",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if not os.path.isfile(args.baseline_data):
        print(f"[ERROR] Baseline data file not found: {args.baseline_data}",
              file=sys.stderr)
        sys.exit(1)

    baseline_records = load_entropy_data(args.baseline_data)
    print(f"Loaded {len(baseline_records)} baseline records.")

    sfa_records = None
    if args.sfa_data is not None:
        if not os.path.isfile(args.sfa_data):
            print(f"[ERROR] SFA data file not found: {args.sfa_data}",
                  file=sys.stderr)
            sys.exit(1)
        sfa_records = load_entropy_data(args.sfa_data)
        print(f"Loaded {len(sfa_records)} SFA records.")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print_summary("Baseline", baseline_records)
    if sfa_records is not None:
        print_summary("SFA", sfa_records)

    # ------------------------------------------------------------------
    # Create figure (1 x 2)
    # ------------------------------------------------------------------
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    plot_panel_a(ax_a, baseline_records, sfa_records)
    plot_panel_b(ax_b, baseline_records, sfa_records)

    fig.suptitle("Figure 5 — Attention Entropy Analysis", fontsize=15,
                 fontweight="bold", y=1.02)
    fig.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    pdf_path = os.path.join(args.output_dir, "fig5_entropy.pdf")
    png_path = os.path.join(args.output_dir, "fig5_entropy.png")

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    print(f"Saved PDF -> {pdf_path}")
    print(f"Saved PNG -> {png_path}")


if __name__ == "__main__":
    main()
