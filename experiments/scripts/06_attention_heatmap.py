"""
Experiment 6: Attention Heatmap Visualization (Figure 3)
========================================================
InternVL3.5 vision encoder의 attention weight를 추출하여
원본 이미지 위에 heatmap으로 오버레이합니다.

Baseline vs SFA-patched 모델의 attention 분포 차이를 시각적으로 비교.

Technical notes:
- InternVL3.5는 기본적으로 flash attention을 사용하므로,
  attention weight 추출을 위해 use_flash_attn = False 설정 필요.
- InternViT-300M: 24 layers, 32x32 patches + 1 CLS token = 1025 tokens,
  16 heads, dim=1024.

Usage:
    conda activate docmllm

    # Extract baseline attention heatmaps
    python experiments/scripts/06_attention_heatmap.py \
        --mode extract \
        --model_type baseline \
        --model_path /NetDisk/j_son/internvl_35 \
        --image /path/to/chart.png \
        --question "What is the value for Haiti?" \
        --layers 20 21 22 23 \
        --output_dir experiments/results/06_heatmap

    # Extract SFA attention heatmaps
    python experiments/scripts/06_attention_heatmap.py \
        --mode extract \
        --model_type sfa \
        --model_path /NetDisk/j_son/internvl_35 \
        --sfa_checkpoint experiments/results/03_sfa_train/checkpoint/sfa_weights.pt \
        --image /path/to/chart.png \
        --question "What is the value for Haiti?" \
        --layers 20 21 22 23 \
        --output_dir experiments/results/06_heatmap

    # Compose baseline + SFA into a single comparison figure
    python experiments/scripts/06_attention_heatmap.py \
        --mode compose \
        --image /path/to/chart.png \
        --output_dir experiments/results/06_heatmap
"""

import argparse
import os
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Matplotlib configuration (non-interactive backend, publication style)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "STIXGeneral"],
    "font.size": 10,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_internvl, load_image_single_tile, run_chat

from importlib import import_module
_sfa_integration = import_module("03_sfa_integration")
patch_internvit_with_sfa = _sfa_integration.patch_internvit_with_sfa


# ============================================================================
# Core helpers
# ============================================================================

def _get_vision_encoder(model):
    """Locate the InternViT vision encoder inside the model."""
    if hasattr(model, "vision_model"):
        return model.vision_model
    if hasattr(model, "model") and hasattr(model.model, "vision_model"):
        return model.model.vision_model
    raise RuntimeError("Could not locate InternViT vision model.")


def disable_flash_attention(model):
    """
    Disable flash attention on every layer of the vision encoder so that
    standard softmax attention is used and weights can be captured via hooks.
    """
    vision_model = _get_vision_encoder(model)
    count = 0
    for layer in vision_model.encoder.layers:
        if hasattr(layer.attn, "use_flash_attn"):
            layer.attn.use_flash_attn = False
            count += 1
    print(f"Disabled flash attention on {count} vision encoder layers.")


def register_attention_hooks(model, target_layers):
    """
    Register forward hooks on the specified vision encoder layers.
    Each hook recomputes the attention weights from the QKV projection
    and stores the CLS-to-spatial attention map (averaged over heads)
    as a 32x32 numpy array.

    Args:
        model: InternVL model.
        target_layers: list of int, layer indices to hook.

    Returns:
        attn_maps: dict that will be populated {layer_idx: np.ndarray[32,32]}
        hooks: list of hook handles (to be removed after the forward pass)
    """
    vision_model = _get_vision_encoder(model)
    encoder_layers = vision_model.encoder.layers
    attn_maps = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach()
            B, N, C = x.shape
            num_heads = module.num_heads
            head_dim = C // num_heads

            # Recompute QKV from the hidden states entering the attention layer
            qkv = module.qkv(x).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)  # each: [B, H, N, d]

            attn_w = (q * (head_dim ** -0.5)) @ k.transpose(-2, -1)  # [B, H, N, N]
            attn_w = attn_w.softmax(dim=-1)

            # CLS token's attention to spatial tokens, averaged over heads
            # CLS is token 0; spatial tokens are 1..N-1
            attn_avg = attn_w.mean(dim=1)[0, 0, 1:]  # [N-1]

            # Reshape to 32x32 patch grid
            grid_size = int(attn_avg.shape[0] ** 0.5)
            attn_maps[layer_idx] = (
                attn_avg[:grid_size * grid_size]
                .reshape(grid_size, grid_size)
                .cpu()
                .float()
                .numpy()
            )
        return hook_fn

    for layer_idx in target_layers:
        if layer_idx < 0 or layer_idx >= len(encoder_layers):
            print(f"[WARN] Layer {layer_idx} out of range (0..{len(encoder_layers)-1}), skipping.")
            continue
        h = encoder_layers[layer_idx].attn.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    return attn_maps, hooks


def overlay_attention_on_image(image_np, attn_map, cmap_name="YlOrRd", alpha=0.5):
    """
    Overlay a 2-D attention map on an RGB image.

    Args:
        image_np: np.ndarray [H, W, 3] uint8 original image.
        attn_map: np.ndarray [h, w] float attention values.
        cmap_name: matplotlib colormap name.
        alpha: blending factor for the heatmap overlay.

    Returns:
        blended: np.ndarray [H, W, 3] float in [0, 1].
    """
    from PIL import Image

    H, W = image_np.shape[:2]

    # Resize attention map to image dimensions
    attn_resized = np.array(
        Image.fromarray(attn_map).resize((W, H), resample=Image.BILINEAR)
    )

    # Normalize to [0, 1]
    vmin, vmax = attn_resized.min(), attn_resized.max()
    if vmax - vmin > 1e-8:
        attn_norm = (attn_resized - vmin) / (vmax - vmin)
    else:
        attn_norm = np.zeros_like(attn_resized)

    # Apply colormap
    cmap = plt.get_cmap(cmap_name)
    heatmap_rgba = cmap(attn_norm)  # [H, W, 4]
    heatmap_rgb = heatmap_rgba[:, :, :3]  # drop alpha channel

    # Blend
    img_float = image_np.astype(np.float64) / 255.0
    blended = (1 - alpha) * img_float + alpha * heatmap_rgb

    return np.clip(blended, 0, 1)


# ============================================================================
# Mode: extract
# ============================================================================

def run_extract(args):
    """
    Extract attention heatmaps from a single image and save per-layer PNGs.
    """
    from PIL import Image

    print(f"=== Attention Heatmap Extraction ({args.model_type}) ===")
    print(f"  Image:    {args.image}")
    print(f"  Question: {args.question}")
    print(f"  Layers:   {args.layers}")
    print()

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    model, tokenizer = load_internvl(args.model_path)

    # ------------------------------------------------------------------
    # 2. Optionally patch with SFA
    # ------------------------------------------------------------------
    if args.model_type == "sfa":
        model = patch_internvit_with_sfa(model, num_patches_h=32, num_patches_w=32)
        if args.sfa_checkpoint and os.path.isfile(args.sfa_checkpoint):
            print(f"Loading SFA checkpoint: {args.sfa_checkpoint}")
            ckpt = torch.load(args.sfa_checkpoint, map_location="cuda")
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                ckpt = ckpt["model_state_dict"]
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print(f"  SFA checkpoint loaded (missing={len(missing)}, unexpected={len(unexpected)})")
        elif args.sfa_checkpoint:
            print(f"[WARN] SFA checkpoint not found: {args.sfa_checkpoint}")

    # ------------------------------------------------------------------
    # 3. Disable flash attention
    # ------------------------------------------------------------------
    disable_flash_attention(model)

    # ------------------------------------------------------------------
    # 4. Register hooks on target layers
    # ------------------------------------------------------------------
    attn_maps, hooks = register_attention_hooks(model, args.layers)

    # ------------------------------------------------------------------
    # 5. Forward pass
    # ------------------------------------------------------------------
    pixel_values = load_image_single_tile(args.image)
    print(f"Running inference (max_new_tokens=16) ...")
    response = run_chat(model, tokenizer, pixel_values, args.question, max_new_tokens=16)
    print(f"  Model response: {response}")

    # Remove hooks
    for h in hooks:
        h.remove()

    if not attn_maps:
        print("[ERROR] No attention maps captured. Check layer indices.")
        return

    # ------------------------------------------------------------------
    # 6. Save heatmap overlays
    # ------------------------------------------------------------------
    original_img = np.array(Image.open(args.image).convert("RGB"))

    out_sub = os.path.join(args.output_dir, args.model_type)
    os.makedirs(out_sub, exist_ok=True)

    for layer_idx, attn_map in sorted(attn_maps.items()):
        blended = overlay_attention_on_image(original_img, attn_map, cmap_name="YlOrRd", alpha=0.5)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(blended)
        ax.set_title(f"{args.model_type.upper()} — Layer {layer_idx}", fontsize=12)
        ax.axis("off")

        fname = f"attn_L{layer_idx:02d}_{args.model_type}.png"
        out_path = os.path.join(out_sub, fname)
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved: {out_path}")

        # Also save raw attention map as .npy for later compose
        npy_path = os.path.join(out_sub, f"attn_L{layer_idx:02d}_{args.model_type}.npy")
        np.save(npy_path, attn_map)

    print(f"\nDone. {len(attn_maps)} heatmap(s) saved to {out_sub}/")


# ============================================================================
# Mode: compose
# ============================================================================

def run_compose(args):
    """
    Load previously saved baseline and SFA heatmaps and combine them into
    a single comparison figure: [Original] [Baseline Attn (L23)] [SFA Attn (L23)].
    """
    from PIL import Image

    print("=== Composing Baseline vs SFA Comparison Figure ===")

    # Determine the layer to use (last in the list, default L23)
    layer_idx = args.layers[-1] if args.layers else 23

    baseline_npy = os.path.join(args.output_dir, "baseline", f"attn_L{layer_idx:02d}_baseline.npy")
    sfa_npy = os.path.join(args.output_dir, "sfa", f"attn_L{layer_idx:02d}_sfa.npy")

    # Check that required files exist
    missing = []
    if not os.path.isfile(baseline_npy):
        missing.append(baseline_npy)
    if not os.path.isfile(sfa_npy):
        missing.append(sfa_npy)
    if missing:
        print("[ERROR] Missing attention map files:")
        for m in missing:
            print(f"  {m}")
        print("Run --mode extract for both --model_type baseline and sfa first.")
        return

    baseline_attn = np.load(baseline_npy)
    sfa_attn = np.load(sfa_npy)

    original_img = np.array(Image.open(args.image).convert("RGB"))

    baseline_overlay = overlay_attention_on_image(original_img, baseline_attn, cmap_name="YlOrRd", alpha=0.5)
    sfa_overlay = overlay_attention_on_image(original_img, sfa_attn, cmap_name="YlOrRd", alpha=0.5)

    # ------------------------------------------------------------------
    # Build 1x3 figure: [Original] [Baseline L23] [SFA L23]
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(baseline_overlay)
    axes[1].set_title(f"Baseline Attention (L{layer_idx})", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(sfa_overlay)
    axes[2].set_title(f"SFA Attention (L{layer_idx})", fontsize=12)
    axes[2].axis("off")

    fig.suptitle("Figure 3: Attention Heatmap Comparison", fontsize=14, y=1.02)
    fig.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)

    pdf_path = os.path.join(args.output_dir, f"figure3_attention_comparison_L{layer_idx}.pdf")
    png_path = os.path.join(args.output_dir, f"figure3_attention_comparison_L{layer_idx}.png")

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    print(f"  Saved PDF: {pdf_path}")
    print(f"  Saved PNG: {png_path}")
    print("Done.")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Figure 3: Attention Heatmap Visualization for InternVL3.5"
    )
    parser.add_argument(
        "--model_type",
        choices=["baseline", "sfa"],
        default="baseline",
        help="Model variant to extract attention from.",
    )
    parser.add_argument(
        "--model_path",
        default="/NetDisk/j_son/internvl_35",
        help="Path to InternVL3.5 model weights.",
    )
    parser.add_argument(
        "--sfa_checkpoint",
        default=None,
        help="Path to SFA fine-tuned weights (only used with --model_type sfa).",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--question",
        default="What is the value for Haiti?",
        help="Query string for the model.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[20, 21, 22, 23],
        help="Layer indices to extract attention from (0-indexed).",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/results/06_heatmap",
        help="Directory to save output heatmaps.",
    )
    parser.add_argument(
        "--mode",
        choices=["extract", "compose"],
        default="extract",
        help=(
            "'extract': run model and save per-layer heatmaps. "
            "'compose': combine saved baseline + SFA heatmaps into one figure."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "extract":
        run_extract(args)
    elif args.mode == "compose":
        run_compose(args)


if __name__ == "__main__":
    main()
