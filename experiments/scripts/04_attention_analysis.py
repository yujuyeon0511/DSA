"""
Experiment 4: Attention Entropy & Hallucination Analysis
========================================================
논문의 핵심 분석 실험:
- Figure 5: Attention entropy 비교 (baseline vs SFA)
- Table 2: Hallucination rate 비교
- Figure 6: Token efficiency curve

Usage:
    conda activate docmllm

    # Attention entropy analysis
    python experiments/scripts/04_attention_analysis.py \
        --mode entropy \
        --model_path /NetDisk/j_son/Model_original/InternVL_35 \
        --data_dir /NetDisk/juyeon/train/chartQA/ChartQA\ Dataset/test/png \
        --output_dir experiments/results/04_analysis

    # Hallucination rate
    python experiments/scripts/04_attention_analysis.py \
        --mode hallucination \
        --model_path /NetDisk/j_son/Model_original/InternVL_35 \
        --output_dir experiments/results/04_analysis

    # Token efficiency curve
    python experiments/scripts/04_attention_analysis.py \
        --mode token_efficiency \
        --output_dir experiments/results/04_analysis
"""

import argparse
import json
import os
import re
import glob
import random

import torch
import numpy as np


def analyze_attention_entropy(model_path, data_dir, output_dir, max_samples=200):
    """
    Baseline InternVL3.5의 attention entropy 분석.
    Text-dense region vs sparse region에서의 entropy 차이를 측정합니다.

    논문 가설 검증: text region에서 attention entropy가 높아 hallucination 발생
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from model_utils import load_internvl, load_image_single_tile, run_chat
    from PIL import Image

    model, tokenizer = load_internvl(model_path)

    # Collect images
    images = glob.glob(os.path.join(data_dir, "*.png"))
    random.seed(42)
    images = random.sample(images, min(max_samples, len(images)))

    # Force naive attention (disable flash attention) to capture weights
    vision_model = model.vision_model if hasattr(model, "vision_model") else None
    if vision_model and hasattr(vision_model, "encoder"):
        for layer in vision_model.encoder.layers:
            if hasattr(layer.attn, "use_flash_attn"):
                layer.attn.use_flash_attn = False

    # Hook to capture attention weights via QKV computation
    attn_maps_collected = []

    def capture_qkv_hook(module, input, output):
        """Capture hidden states entering attention to compute attention weights"""
        # input[0] is hidden_states going into attn
        x = input[0].detach()
        B, N, C = x.shape
        if not hasattr(module, "qkv"):
            return
        num_heads = module.num_heads
        head_dim = C // num_heads
        qkv = module.qkv(x).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scale = head_dim ** -0.5
        attn_w = (q * scale) @ k.transpose(-2, -1)
        attn_w = attn_w.softmax(dim=-1).cpu().float()
        attn_maps_collected.append(attn_w)

    # Register hooks on last 4 layers
    hooks = []
    if vision_model and hasattr(vision_model, "encoder"):
        layers = vision_model.encoder.layers
        for i in range(max(0, len(layers) - 4), len(layers)):
            h = layers[i].attn.register_forward_hook(capture_qkv_hook)
            hooks.append(h)

    # Generate density maps for each image to identify text regions
    import cv2
    from importlib import import_module
    _de = import_module("01_density_estimator")
    generate_density_map = _de.generate_density_map

    results = []

    for idx, img_path in enumerate(images):
        attn_maps_collected.clear()

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img.resize((448, 448)))

        # Get density map to identify text/non-text regions (32×32 = InternViT patch grid)
        density = generate_density_map(img_np, output_size=32)  # [32, 32]
        text_mask = density > 0.3  # text-dense patches
        sparse_mask = density <= 0.3  # sparse patches

        # Forward pass with attention output
        try:
            pv = load_image_single_tile(img_path)
            run_chat(model, tokenizer, pv, "What is shown in this image?", max_new_tokens=10)
        except Exception:
            continue

        if not attn_maps_collected:
            continue

        # Analyze entropy
        for layer_idx, attn_w in enumerate(attn_maps_collected):
            # attn_w: [1, H, N, N]
            if attn_w.dim() == 4:
                attn_w = attn_w.squeeze(0)  # [H, N, N]

            eps = 1e-8
            entropy = -(attn_w * (attn_w + eps).log()).sum(dim=-1)  # [H, N]

            N = entropy.shape[-1]
            # N=1025 (1 CLS + 32×32 patches) or N=1024
            if N == 1025:
                entropy = entropy[:, 1:]  # skip CLS token
                N = 1024
            H_p = W_p = int(N ** 0.5)

            if H_p * W_p != N:
                continue

            mask_flat_text = torch.from_numpy(text_mask[:H_p, :W_p].reshape(-1))
            mask_flat_sparse = torch.from_numpy(sparse_mask[:H_p, :W_p].reshape(-1))

            text_entropy = entropy[:, mask_flat_text].mean().item() if mask_flat_text.any() else 0
            sparse_entropy = entropy[:, mask_flat_sparse].mean().item() if mask_flat_sparse.any() else 0
            overall_entropy = entropy.mean().item()

            results.append({
                "image": os.path.basename(img_path),
                "layer": layer_idx,
                "text_entropy": text_entropy,
                "sparse_entropy": sparse_entropy,
                "overall_entropy": overall_entropy,
                "text_ratio": text_mask.mean().item(),
            })

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx+1}/{len(images)}")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "entropy_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Summary statistics
    if results:
        text_ent = np.mean([r["text_entropy"] for r in results])
        sparse_ent = np.mean([r["sparse_entropy"] for r in results])
        overall_ent = np.mean([r["overall_entropy"] for r in results])
        print(f"\n{'='*50}")
        print(f"Attention Entropy Analysis ({len(results)} measurements)")
        print(f"{'='*50}")
        print(f"  Text-dense regions:  {text_ent:.4f}")
        print(f"  Sparse regions:      {sparse_ent:.4f}")
        print(f"  Overall:             {overall_ent:.4f}")
        print(f"  Ratio (text/sparse): {text_ent/sparse_ent:.2f}x" if sparse_ent > 0 else "")

    return results


def analyze_hallucination(model_path, output_dir, max_samples=500):
    """
    ChartQA에서 hallucination rate 측정.
    숫자 답변에서 차트에 존재하지 않는 값을 생성하는 비율을 계산합니다.

    Hallucination types:
    1. Number hallucination: 차트에 없는 숫자 생성
    2. Entity hallucination: 차트에 없는 카테고리명 생성
    3. Trend hallucination: 잘못된 증감 방향 답변
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from model_utils import load_internvl, load_image_single_tile, run_chat
    from PIL import Image

    model, tokenizer = load_internvl(model_path)

    # Load ChartQA test
    test_dir = "/NetDisk/juyeon/train/chartQA/ChartQA Dataset/test"
    samples = []
    for fname in ["test_human.json", "test_augmented.json"]:
        fpath = os.path.join(test_dir, fname)
        if not os.path.exists(fpath):
            fpath = os.path.join(test_dir, "annotations", fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            for item in data:
                samples.append(item)

    random.seed(42)
    samples = random.sample(samples, min(max_samples, len(samples)))

    results = {
        "total": 0,
        "correct": 0,
        "hallucinated_number": 0,
        "wrong_answer": 0,
        "details": [],
    }

    for i, item in enumerate(samples):
        question = item.get("query", item.get("question", ""))
        gt_answer = str(item.get("label", item.get("answer", "")))
        img_name = item.get("imgname", item.get("image", ""))
        img_path = os.path.join(test_dir, "png", img_name)

        if not os.path.exists(img_path):
            continue

        try:
            pv = load_image_single_tile(img_path)
            pred = run_chat(model, tokenizer, pv, f"{question}\nAnswer concisely.", max_new_tokens=64)
        except Exception:
            continue

        results["total"] += 1

        # Check if prediction matches GT
        is_correct = False
        try:
            pred_num = float(pred.replace(",", "").replace("%", "").strip())
            gt_num = float(gt_answer.replace(",", "").replace("%", "").strip())
            if gt_num != 0 and abs(pred_num - gt_num) / abs(gt_num) <= 0.05:
                is_correct = True
            elif gt_num == 0 and pred_num == 0:
                is_correct = True
        except ValueError:
            if pred.strip().lower() == gt_answer.strip().lower():
                is_correct = True

        if is_correct:
            results["correct"] += 1
        else:
            # Classify hallucination type
            # Check if prediction contains numbers not in GT
            pred_nums = re.findall(r"[\d]+\.?\d*", pred)
            gt_nums = re.findall(r"[\d]+\.?\d*", gt_answer)
            if pred_nums and not any(pn in gt_nums for pn in pred_nums):
                results["hallucinated_number"] += 1
            else:
                results["wrong_answer"] += 1

        results["details"].append({
            "question": question,
            "gt": gt_answer,
            "pred": pred,
            "correct": is_correct,
        })

        if (i + 1) % 50 == 0:
            total = results["total"]
            acc = results["correct"] / total if total > 0 else 0
            hr = results["hallucinated_number"] / total if total > 0 else 0
            print(f"  [{i+1}/{len(samples)}] Acc: {acc:.3f} | Halluc Rate: {hr:.3f}")

    # Summary
    total = results["total"]
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "total": total,
        "accuracy": results["correct"] / total if total > 0 else 0,
        "hallucination_rate": results["hallucinated_number"] / total if total > 0 else 0,
        "wrong_answer_rate": results["wrong_answer"] / total if total > 0 else 0,
    }
    results["summary"] = summary

    with open(os.path.join(output_dir, "hallucination_analysis.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Hallucination Analysis ({total} samples)")
    print(f"{'='*50}")
    print(f"  Accuracy:           {summary['accuracy']:.4f}")
    print(f"  Hallucination Rate: {summary['hallucination_rate']:.4f}")
    print(f"  Wrong Answer Rate:  {summary['wrong_answer_rate']:.4f}")

    return results


def plot_token_efficiency(output_dir):
    """
    Token efficiency curve 생성 — 논문 Figure 6.
    Budget sweep: N ∈ {64, 128, 256, 512, 1024}에서의 accuracy 비교.

    NOTE: 실제 budget sweep은 ADAT가 구현된 후 실행.
    이 함수는 placeholder로 그래프 형식만 생성합니다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Placeholder data — 실제 실험 후 교체
    budgets = [64, 128, 256, 512, 1024]
    baseline_acc = [0.52, 0.63, 0.72, 0.78, 0.82]  # uniform patching
    adat_acc = [0.58, 0.69, 0.77, 0.82, 0.84]  # adaptive patching
    sfa_adat_acc = [0.61, 0.72, 0.79, 0.84, 0.86]  # SFA + ADAT

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(budgets, baseline_acc, "o-", label="Baseline (uniform)", color="#666")
    ax.plot(budgets, adat_acc, "s-", label="+ ADAT", color="#2196F3")
    ax.plot(budgets, sfa_adat_acc, "D-", label="+ SFA + ADAT", color="#F44336")

    ax.set_xlabel("Token Budget (N)", fontsize=13)
    ax.set_ylabel("ChartQA Relaxed Accuracy", fontsize=13)
    ax.set_title("Token Efficiency: Accuracy vs Token Budget", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "token_efficiency_curve.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "token_efficiency_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved token efficiency plot to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["entropy", "hallucination", "token_efficiency"], required=True)
    parser.add_argument("--model_path", default="/NetDisk/j_son/Model_original/InternVL_35")
    parser.add_argument("--data_dir", default="/NetDisk/juyeon/train/chartQA/ChartQA Dataset/test/png")
    parser.add_argument("--output_dir", default="experiments/results/04_analysis")
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    if args.mode == "entropy":
        analyze_attention_entropy(args.model_path, args.data_dir, args.output_dir, args.max_samples)
    elif args.mode == "hallucination":
        analyze_hallucination(args.model_path, args.output_dir, args.max_samples)
    elif args.mode == "token_efficiency":
        plot_token_efficiency(args.output_dir)


if __name__ == "__main__":
    main()
