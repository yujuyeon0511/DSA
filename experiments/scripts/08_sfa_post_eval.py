"""
Post-SFA-Training Comprehensive Evaluation
============================================
모델을 한 번 로드하여 모든 후속 분석을 수행합니다:
  1. ChartQA eval (P2-3) → Table 1 "+SFA" accuracy
  2. Hallucination analysis (P2-5) → Table 2 comparison
  3. Entropy analysis (P2-4) → entropy_analysis_sfa.json

Usage:
    conda activate docmllm
    cd /NetDisk/juyeon/DSA

    python -u experiments/scripts/08_sfa_post_eval.py \
        --model_path /NetDisk/j_son/internvl_35 \
        --sfa_checkpoint experiments/results/03_sfa_train/best.pth \
        --output_dir experiments/results/05_sfa_eval
"""

import argparse
import glob
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_internvl, load_image_single_tile, run_chat

from importlib import import_module
_sfa_int = import_module("03_sfa_integration")
patch_internvit_with_sfa = _sfa_int.patch_internvit_with_sfa

_sfa_ft = import_module("03_sfa_finetune")
load_sfa_weights = _sfa_ft.load_sfa_weights


# ─── Model loading ─────────────────────────────────────────

def load_sfa_model(model_path, sfa_checkpoint, device="cuda", dtype=torch.bfloat16):
    """Load base model, patch with SFA, load fine-tuned weights."""
    print("=" * 60)
    print("Loading SFA model for post-training evaluation")
    print("=" * 60)

    model, tokenizer = load_internvl(model_path, device=device, dtype=dtype)
    model = patch_internvit_with_sfa(model)

    if sfa_checkpoint and os.path.exists(sfa_checkpoint):
        load_sfa_weights(model, sfa_checkpoint)
    else:
        print(f"[WARN] SFA checkpoint not found: {sfa_checkpoint}")

    model.eval()

    mem = torch.cuda.memory_allocated(device) / 1024**3
    print(f"  GPU memory: {mem:.1f} GB")
    return model, tokenizer


# ─── P2-3 + P2-5: ChartQA Eval + Hallucination ───────────

def _relaxed_match(pred, gt, tol=0.05):
    pred, gt = pred.strip().lower(), gt.strip().lower()
    if pred == gt:
        return True
    try:
        pn = float(pred.replace(",", "").replace("%", ""))
        gn = float(gt.replace(",", "").replace("%", ""))
        if gn == 0:
            return pn == 0
        return abs(pn - gn) / abs(gn) <= tol
    except ValueError:
        return False


def run_chartqa_eval(model, tokenizer, test_dir, output_dir, max_samples=None):
    """
    ChartQA test eval + hallucination classification.
    Returns (accuracy, halluc_rate, results_dict).
    """
    print("\n" + "=" * 60)
    print("P2-3 + P2-5: ChartQA Evaluation + Hallucination Analysis")
    print("=" * 60)

    samples = []
    for fname in ["test_human.json", "test_augmented.json"]:
        fpath = os.path.join(test_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                samples.extend(json.load(f))

    if max_samples:
        random.seed(42)
        samples = random.sample(samples, min(max_samples, len(samples)))

    print(f"  Test samples: {len(samples)}")

    results = {
        "total": 0, "correct": 0,
        "hallucinated_number": 0, "wrong_answer": 0,
        "details": [],
    }
    t0 = time.time()

    for i, item in enumerate(samples):
        question = item.get("query", item.get("question", ""))
        gt = str(item.get("label", item.get("answer", "")))
        img_name = item.get("imgname", item.get("image", ""))
        img_path = os.path.join(test_dir, "png", img_name)

        if not os.path.exists(img_path):
            continue

        try:
            pv = load_image_single_tile(img_path)
            pred = run_chat(model, tokenizer, pv,
                            f"{question}\nAnswer concisely.", max_new_tokens=64)
        except Exception:
            pred = ""

        results["total"] += 1
        correct = _relaxed_match(pred, gt)

        if correct:
            results["correct"] += 1
            error_type = "correct"
        else:
            pred_nums = re.findall(r"[\d]+\.?\d*", pred)
            gt_nums = re.findall(r"[\d]+\.?\d*", gt)
            if pred_nums and not any(pn in gt_nums for pn in pred_nums):
                results["hallucinated_number"] += 1
                error_type = "hallucinated_number"
            else:
                results["wrong_answer"] += 1
                error_type = "wrong_answer"

        results["details"].append({
            "question": question, "gt": gt, "pred": pred,
            "correct": correct, "error_type": error_type,
        })

        if (i + 1) % 50 == 0:
            total = results["total"]
            acc = results["correct"] / total
            hr = results["hallucinated_number"] / total
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(samples)}] Acc: {acc:.4f} | Halluc: {hr:.4f} | {elapsed:.0f}s")

    total = results["total"]
    elapsed = time.time() - t0
    results["accuracy"] = results["correct"] / max(total, 1)
    results["halluc_rate"] = results["hallucinated_number"] / max(total, 1)
    results["wrong_rate"] = results["wrong_answer"] / max(total, 1)
    results["time_sec"] = elapsed

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  {'='*50}")
    print(f"  ChartQA SFA Evaluation Results")
    print(f"  {'='*50}")
    print(f"  Total samples:      {total}")
    print(f"  Relaxed Accuracy:   {results['accuracy']:.4f}")
    print(f"  Hallucination Rate: {results['halluc_rate']:.4f}")
    print(f"  Wrong Answer Rate:  {results['wrong_rate']:.4f}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  Saved to {output_dir}/eval_results.json")

    return results


# ─── P2-4: Entropy Analysis ─────────────────────────────────

def run_entropy_analysis(model, tokenizer, data_dir, output_dir, max_samples=200):
    """
    SFA model의 attention entropy 분석.
    SFAAttention._last_attn_weights를 사용하여 structural bias 포함된 attention의 entropy 측정.
    """
    print("\n" + "=" * 60)
    print("P2-4: Attention Entropy Analysis (SFA)")
    print("=" * 60)

    from PIL import Image

    # Density map for region classification
    _de = import_module("01_density_estimator")
    generate_density_map = _de.generate_density_map

    images = glob.glob(os.path.join(data_dir, "*.png"))
    random.seed(42)
    images = random.sample(images, min(max_samples, len(images)))
    print(f"  Analyzing {len(images)} images")

    vision_model = model.vision_model
    n_layers = len(vision_model.encoder.layers)
    target_layers = list(range(max(0, n_layers - 4), n_layers))
    print(f"  Target layers: {target_layers}")

    results = []
    t0 = time.time()

    for idx, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img.resize((448, 448)))
        density = generate_density_map(img_np, output_size=32)
        text_mask = density > 0.3
        sparse_mask = density <= 0.3

        try:
            pv = load_image_single_tile(img_path)
            with torch.no_grad():
                run_chat(model, tokenizer, pv,
                         "What is shown in this image?", max_new_tokens=10)
        except Exception:
            continue

        for li, layer_idx in enumerate(target_layers):
            layer = vision_model.encoder.layers[layer_idx]
            attn_w = getattr(layer.attn, '_last_attn_weights', None)
            if attn_w is None:
                continue

            # attn_w: [B, H, N, N] or [H, N, N]
            if attn_w.dim() == 4:
                attn_w = attn_w.squeeze(0)

            eps = 1e-8
            entropy = -(attn_w * (attn_w + eps).log()).sum(dim=-1)  # [H, N]

            N = entropy.shape[-1]
            if N == 1025:
                entropy = entropy[:, 1:]
                N = 1024
            H_p = W_p = int(N ** 0.5)
            if H_p * W_p != N:
                continue

            mask_text = torch.from_numpy(
                text_mask[:H_p, :W_p].reshape(-1)
            ).to(entropy.device)
            mask_sparse = torch.from_numpy(
                sparse_mask[:H_p, :W_p].reshape(-1)
            ).to(entropy.device)

            text_ent = entropy[:, mask_text].mean().item() if mask_text.any() else 0
            sparse_ent = entropy[:, mask_sparse].mean().item() if mask_sparse.any() else 0
            overall_ent = entropy.mean().item()

            results.append({
                "image": os.path.basename(img_path),
                "layer": li,
                "text_entropy": text_ent,
                "sparse_entropy": sparse_ent,
                "overall_entropy": overall_ent,
                "text_ratio": text_mask.mean().item(),
            })

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(images)}] {elapsed:.0f}s")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "entropy_analysis_sfa.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    if results:
        text_ent = np.mean([r["text_entropy"] for r in results])
        sparse_ent = np.mean([r["sparse_entropy"] for r in results])
        overall_ent = np.mean([r["overall_entropy"] for r in results])
        elapsed = time.time() - t0
        print(f"\n  {'='*50}")
        print(f"  SFA Entropy Analysis ({len(results)} measurements)")
        print(f"  {'='*50}")
        print(f"  Text-dense regions:  {text_ent:.4f}")
        print(f"  Sparse regions:      {sparse_ent:.4f}")
        print(f"  Overall:             {overall_ent:.4f}")
        if sparse_ent > 0:
            print(f"  Ratio (text/sparse): {text_ent/sparse_ent:.2f}x")
        print(f"  Time: {elapsed:.0f}s")
        print(f"  Saved to {out_path}")

    return results


# ─── P2-5: Hallucination subset analysis ─────────────────

def run_hallucination_subset(model, tokenizer, test_dir, output_dir, max_samples=200):
    """
    200-sample hallucination analysis (same seed as baseline for fair comparison).
    """
    print("\n" + "=" * 60)
    print("P2-5: Hallucination Subset Analysis (200 samples)")
    print("=" * 60)

    samples = []
    for fname in ["test_human.json", "test_augmented.json"]:
        fpath = os.path.join(test_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                samples.extend(json.load(f))

    random.seed(42)
    samples = random.sample(samples, min(max_samples, len(samples)))
    print(f"  Samples: {len(samples)}")

    results = {
        "total": 0, "correct": 0,
        "hallucinated_number": 0, "wrong_answer": 0,
        "details": [],
    }
    t0 = time.time()

    for i, item in enumerate(samples):
        question = item.get("query", item.get("question", ""))
        gt = str(item.get("label", item.get("answer", "")))
        img_name = item.get("imgname", item.get("image", ""))
        img_path = os.path.join(test_dir, "png", img_name)

        if not os.path.exists(img_path):
            continue

        try:
            pv = load_image_single_tile(img_path)
            pred = run_chat(model, tokenizer, pv,
                            f"{question}\nAnswer concisely.", max_new_tokens=64)
        except Exception:
            pred = ""

        results["total"] += 1
        correct = _relaxed_match(pred, gt)

        if correct:
            results["correct"] += 1
            error_type = "correct"
        else:
            pred_nums = re.findall(r"[\d]+\.?\d*", pred)
            gt_nums = re.findall(r"[\d]+\.?\d*", gt)
            if pred_nums and not any(pn in gt_nums for pn in pred_nums):
                results["hallucinated_number"] += 1
                error_type = "hallucinated_number"
            else:
                results["wrong_answer"] += 1
                error_type = "wrong_answer"

        results["details"].append({
            "question": question, "gt": gt, "pred": pred,
            "correct": correct, "error_type": error_type,
        })

        if (i + 1) % 50 == 0:
            total = results["total"]
            acc = results["correct"] / total
            print(f"  [{i+1}/{len(samples)}] Acc: {acc:.4f}")

    total = results["total"]
    elapsed = time.time() - t0
    results["summary"] = {
        "total": total,
        "accuracy": results["correct"] / max(total, 1),
        "hallucination_rate": results["hallucinated_number"] / max(total, 1),
        "wrong_answer_rate": results["wrong_answer"] / max(total, 1),
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "hallucination_sfa.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    s = results["summary"]
    print(f"\n  {'='*50}")
    print(f"  SFA Hallucination Analysis ({total} samples)")
    print(f"  {'='*50}")
    print(f"  Accuracy:           {s['accuracy']:.4f}")
    print(f"  Hallucination Rate: {s['hallucination_rate']:.4f}")
    print(f"  Wrong Answer Rate:  {s['wrong_answer_rate']:.4f}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  Saved to {out_path}")

    return results


# ─── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFA Post-Training Evaluation")
    parser.add_argument("--model_path", default="/NetDisk/j_son/internvl_35")
    parser.add_argument("--sfa_checkpoint",
                        default="experiments/results/03_sfa_train/best.pth")
    parser.add_argument("--test_dir",
                        default="/NetDisk/juyeon/train/chartQA/ChartQA Dataset/test")
    parser.add_argument("--data_dir",
                        default="/NetDisk/juyeon/train/chartQA/ChartQA Dataset/test/png")
    parser.add_argument("--output_dir",
                        default="experiments/results/05_sfa_eval")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Limit ChartQA eval samples (None=full test set)")
    parser.add_argument("--max_entropy_samples", type=int, default=200)
    parser.add_argument("--max_halluc_samples", type=int, default=200)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_entropy", action="store_true")
    parser.add_argument("--skip_halluc", action="store_true")
    args = parser.parse_args()

    # Load model once
    model, tokenizer = load_sfa_model(
        args.model_path, args.sfa_checkpoint
    )

    summary = {}

    # P2-3: ChartQA eval
    if not args.skip_eval:
        eval_results = run_chartqa_eval(
            model, tokenizer, args.test_dir, args.output_dir,
            max_samples=args.max_eval_samples,
        )
        summary["chartqa_accuracy"] = eval_results["accuracy"]
        summary["chartqa_halluc_rate"] = eval_results["halluc_rate"]

    # P2-5: Hallucination subset (200 samples, same as baseline)
    if not args.skip_halluc:
        halluc_results = run_hallucination_subset(
            model, tokenizer, args.test_dir, args.output_dir,
            max_samples=args.max_halluc_samples,
        )
        summary["halluc_200_accuracy"] = halluc_results["summary"]["accuracy"]
        summary["halluc_200_rate"] = halluc_results["summary"]["hallucination_rate"]

    # P2-4: Entropy analysis
    if not args.skip_entropy:
        entropy_results = run_entropy_analysis(
            model, tokenizer, args.data_dir, args.output_dir,
            max_samples=args.max_entropy_samples,
        )
        if entropy_results:
            text_ent = np.mean([r["text_entropy"] for r in entropy_results])
            sparse_ent = np.mean([r["sparse_entropy"] for r in entropy_results])
            summary["entropy_text"] = text_ent
            summary["entropy_sparse"] = sparse_ent

    # Save summary
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("ALL POST-TRAINING EVALUATION COMPLETE")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
