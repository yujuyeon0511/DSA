"""
Attention Heatmap: Baseline vs SFA (Figure 3)
===============================================
한 번의 실행으로 baseline과 SFA 모델의 attention heatmap을 순차 추출한 뒤
비교 figure를 생성합니다.

Usage:
    python experiments/scripts/11_heatmap_both.py \
        --image experiments/figures/sample_images/chartqa_sample.png \
        --question "What is the value for Haiti?"
"""

import argparse
import gc
import os
import sys
import time

import numpy as np
import torch

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

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_internvl, load_image_single_tile, run_chat
from importlib import import_module

_sfa_integration = import_module("03_sfa_integration")
patch_internvit_with_sfa = _sfa_integration.patch_internvit_with_sfa


def get_vision_encoder(model):
    if hasattr(model, "vision_model"):
        return model.vision_model
    if hasattr(model, "model") and hasattr(model.model, "vision_model"):
        return model.model.vision_model
    raise RuntimeError("Could not locate InternViT vision model.")


def disable_flash_attention(model):
    vision_model = get_vision_encoder(model)
    count = 0
    for layer in vision_model.encoder.layers:
        if hasattr(layer.attn, "use_flash_attn"):
            layer.attn.use_flash_attn = False
            count += 1
    print(f"  Disabled flash attention on {count} layers")


def extract_attention_maps(model, pixel_values, tokenizer, question, target_layers, device="cuda:1"):
    """Extract CLS-to-spatial attention maps from target layers."""
    vision_model = get_vision_encoder(model)
    encoder_layers = vision_model.encoder.layers
    attn_maps = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach()
            B, N, C = x.shape
            num_heads = module.num_heads
            head_dim = C // num_heads

            qkv = module.qkv(x).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)

            attn_w = (q * (head_dim ** -0.5)) @ k.transpose(-2, -1)

            # If SFA, add structural bias
            if hasattr(module, "structural_bias"):
                bias = module.structural_bias()
                attn_w[:, :, 1:, 1:] = attn_w[:, :, 1:, 1:] + bias

            attn_w = attn_w.softmax(dim=-1)

            # CLS to spatial, averaged over heads
            attn_avg = attn_w.mean(dim=1)[0, 0, 1:]
            grid_size = int(attn_avg.shape[0] ** 0.5)
            attn_maps[layer_idx] = (
                attn_avg[:grid_size * grid_size]
                .reshape(grid_size, grid_size)
                .cpu().float().numpy()
            )
        return hook_fn

    for layer_idx in target_layers:
        if layer_idx < len(encoder_layers):
            h = encoder_layers[layer_idx].attn.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

    response = run_chat(model, tokenizer, pixel_values, question, max_new_tokens=16, device=device)
    print(f"  Response: {response}")

    for h in hooks:
        h.remove()

    return attn_maps, response


def overlay_attention(image_np, attn_map, cmap_name="YlOrRd", alpha=0.5):
    from PIL import Image
    H, W = image_np.shape[:2]
    attn_resized = np.array(
        Image.fromarray(attn_map).resize((W, H), resample=Image.BILINEAR)
    )
    vmin, vmax = attn_resized.min(), attn_resized.max()
    if vmax - vmin > 1e-8:
        attn_norm = (attn_resized - vmin) / (vmax - vmin)
    else:
        attn_norm = np.zeros_like(attn_resized)

    cmap = plt.get_cmap(cmap_name)
    heatmap_rgb = cmap(attn_norm)[:, :, :3]
    img_float = image_np.astype(np.float64) / 255.0
    return np.clip((1 - alpha) * img_float + alpha * heatmap_rgb, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/NetDisk/j_son/internvl_35")
    parser.add_argument("--sfa_checkpoint", default="experiments/results/03_sfa_train/best.pth")
    parser.add_argument("--image", default="experiments/figures/sample_images/chartqa_sample.png")
    parser.add_argument("--question", default="What is the value for Haiti?")
    parser.add_argument("--layers", type=int, nargs="+", default=[11, 17, 20, 23])
    parser.add_argument("--output_dir", default="experiments/figures/fig3_attention_heatmap")
    parser.add_argument("--device", default="cuda:1", help="GPU device to use")
    args = parser.parse_args()

    from PIL import Image
    original_img = np.array(Image.open(args.image).convert("RGB"))
    os.makedirs(args.output_dir, exist_ok=True)

    pixel_values = load_image_single_tile(args.image)

    # ================================================================
    # 1. Baseline extraction
    # ================================================================
    print("=" * 60)
    print("PHASE 1: Baseline Attention Extraction")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_internvl(args.model_path, device=args.device)
    disable_flash_attention(model)

    baseline_maps, baseline_resp = extract_attention_maps(
        model, pixel_values, tokenizer, args.question, args.layers, device=args.device
    )
    print(f"  Baseline done in {time.time()-t0:.0f}s")

    # Save baseline maps
    for layer_idx, amap in baseline_maps.items():
        np.save(os.path.join(args.output_dir, f"baseline_L{layer_idx:02d}.npy"), amap)

    # Free memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU memory freed")

    # ================================================================
    # 2. SFA extraction
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: SFA Attention Extraction")
    print("=" * 60)
    t1 = time.time()

    model, tokenizer = load_internvl(args.model_path, device=args.device)
    model = patch_internvit_with_sfa(model, num_patches_h=32, num_patches_w=32)

    if os.path.isfile(args.sfa_checkpoint):
        print(f"  Loading SFA checkpoint: {args.sfa_checkpoint}")
        ckpt = torch.load(args.sfa_checkpoint, map_location=args.device)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"  Loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print(f"  [WARN] Checkpoint not found: {args.sfa_checkpoint}")

    disable_flash_attention(model)

    sfa_maps, sfa_resp = extract_attention_maps(
        model, pixel_values, tokenizer, args.question, args.layers, device=args.device
    )
    print(f"  SFA done in {time.time()-t1:.0f}s")

    for layer_idx, amap in sfa_maps.items():
        np.save(os.path.join(args.output_dir, f"sfa_L{layer_idx:02d}.npy"), amap)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ================================================================
    # 3. Compose comparison figures
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Composing Comparison Figures")
    print("=" * 60)

    # Per-layer comparison: [Original] [Baseline] [SFA]
    for layer_idx in args.layers:
        if layer_idx not in baseline_maps or layer_idx not in sfa_maps:
            continue

        baseline_overlay = overlay_attention(original_img, baseline_maps[layer_idx])
        sfa_overlay = overlay_attention(original_img, sfa_maps[layer_idx])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_img)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(baseline_overlay)
        axes[1].set_title(f"Baseline (Layer {layer_idx})", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(sfa_overlay)
        axes[2].set_title(f"+SFA (Layer {layer_idx})", fontsize=12)
        axes[2].axis("off")

        fig.suptitle(f"Figure 3: Attention Heatmap — Layer {layer_idx}", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f"fig3_L{layer_idx:02d}.pdf"))
        fig.savefig(os.path.join(args.output_dir, f"fig3_L{layer_idx:02d}.png"))
        plt.close(fig)

    # Multi-layer grid: 2 rows (Baseline/SFA) × N cols (layers)
    n_layers = len(args.layers)
    fig, axes = plt.subplots(2, n_layers + 1, figsize=(4 * (n_layers + 1), 8))

    # Original image in first column
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original", fontsize=11)
    axes[0, 0].axis("off")
    axes[1, 0].imshow(original_img)
    axes[1, 0].set_title("Original", fontsize=11)
    axes[1, 0].axis("off")

    for col, layer_idx in enumerate(args.layers, start=1):
        if layer_idx in baseline_maps:
            axes[0, col].imshow(overlay_attention(original_img, baseline_maps[layer_idx]))
            axes[0, col].set_title(f"Baseline L{layer_idx}", fontsize=11)
        axes[0, col].axis("off")

        if layer_idx in sfa_maps:
            axes[1, col].imshow(overlay_attention(original_img, sfa_maps[layer_idx]))
            axes[1, col].set_title(f"+SFA L{layer_idx}", fontsize=11)
        axes[1, col].axis("off")

    # Row labels
    axes[0, 0].set_ylabel("Baseline", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("+SFA", fontsize=13, fontweight="bold")

    fig.suptitle(
        f"Figure 3: Attention Heatmap Comparison (Baseline vs SFA)\n"
        f"Q: \"{args.question}\"  |  Baseline: \"{baseline_resp}\"  |  SFA: \"{sfa_resp}\"",
        fontsize=12, y=1.03
    )
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "fig3_attention_comparison.pdf"))
    fig.savefig(os.path.join(args.output_dir, "fig3_attention_comparison.png"))
    plt.close(fig)

    print(f"\nFigure 3 saved to {args.output_dir}/")
    print(f"  fig3_attention_comparison.png — Multi-layer grid comparison")
    for layer_idx in args.layers:
        print(f"  fig3_L{layer_idx:02d}.png — Layer {layer_idx} detail")
    print(f"\n  Baseline response: {baseline_resp}")
    print(f"  SFA response:      {sfa_resp}")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
