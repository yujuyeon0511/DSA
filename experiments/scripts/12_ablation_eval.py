"""
Ablation Study Evaluation (P2-8)
=================================
SFA-only 학습 결과를 평가합니다.
ChartQA eval + Hallucination subset + Entropy 분석을 한 번에 실행.

Usage:
    python -u experiments/scripts/12_ablation_eval.py \
        --sfa_checkpoint experiments/results/06_ablation_sfa_only/best.pth \
        --output_dir experiments/results/06_ablation_sfa_only/eval \
        --label "SFA-only (7,296 params)"
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_internvl, load_image_single_tile, run_chat
from importlib import import_module
_sfa_int = import_module("03_sfa_integration")
patch_internvit_with_sfa = _sfa_int.patch_internvit_with_sfa


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


def _classify_hallucination(pred, gt, correct):
    if correct:
        return "correct"
    pred_s, gt_s = pred.strip(), gt.strip()
    try:
        float(gt_s.replace(",", "").replace("%", ""))
        try:
            float(pred_s.replace(",", "").replace("%", ""))
            return "number_hallucination"
        except ValueError:
            pass
    except ValueError:
        pass
    return "wrong_other"


def load_sfa_model(model_path, sfa_checkpoint, device="cuda"):
    """Load model with SFA and checkpoint."""
    dtype = torch.bfloat16
    model, tokenizer = load_internvl(model_path, device=device, dtype=dtype)
    model = patch_internvit_with_sfa(model, num_patches_h=32, num_patches_w=32)

    if sfa_checkpoint and os.path.isfile(sfa_checkpoint):
        print(f"Loading SFA checkpoint: {sfa_checkpoint}")
        ckpt = torch.load(sfa_checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"  Loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    model.eval()
    return model, tokenizer


def run_chartqa_eval(model, tokenizer, test_dir, max_samples=2500):
    """Run ChartQA evaluation."""
    print("\n" + "=" * 50)
    print("ChartQA Evaluation")
    print("=" * 50)

    samples = []
    for fname in ["test_human.json", "test_augmented.json"]:
        fpath = os.path.join(test_dir, fname)
        if not os.path.exists(fpath):
            fpath = os.path.join(test_dir, "annotations", fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            for item in data:
                samples.append({
                    "imgname": item.get("imgname", item.get("image", "")),
                    "query": item.get("query", item.get("question", "")),
                    "label": str(item.get("label", item.get("answer", ""))),
                })

    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]

    results = {"total": 0, "correct": 0, "hallucinations": 0, "details": []}
    t0 = time.time()

    for i, sample in enumerate(samples):
        img_path = os.path.join(test_dir, "png", sample["imgname"])
        if not os.path.exists(img_path):
            continue

        try:
            pv = load_image_single_tile(img_path)
            pred = run_chat(model, tokenizer, pv,
                           f"{sample['query']}\nAnswer concisely.", max_new_tokens=64)
        except Exception:
            pred = ""

        gt = sample["label"]
        correct = _relaxed_match(pred, gt)
        h_type = _classify_hallucination(pred, gt, correct)

        results["total"] += 1
        if correct:
            results["correct"] += 1
        if h_type == "number_hallucination":
            results["hallucinations"] += 1

        results["details"].append({
            "question": sample["query"], "gt": gt, "pred": pred,
            "correct": correct, "halluc_type": h_type,
        })

        if (i + 1) % 100 == 0:
            acc = results["correct"] / results["total"]
            h_rate = results["hallucinations"] / results["total"]
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(samples)}] Acc: {acc:.4f} | Halluc: {h_rate:.4f} | {elapsed:.0f}s")

    acc = results["correct"] / max(results["total"], 1)
    h_rate = results["hallucinations"] / max(results["total"], 1)
    results["accuracy"] = acc
    results["halluc_rate"] = h_rate

    print(f"\n  ChartQA Acc: {acc:.4f}, Halluc Rate: {h_rate:.4f}")
    return results


def run_hallucination_subset(model, tokenizer, test_dir, n=200):
    """Run hallucination analysis on fixed 200-sample subset."""
    print("\n" + "=" * 50)
    print(f"Hallucination Subset ({n} samples)")
    print("=" * 50)

    samples = []
    for fname in ["test_human.json", "test_augmented.json"]:
        fpath = os.path.join(test_dir, fname)
        if not os.path.exists(fpath):
            fpath = os.path.join(test_dir, "annotations", fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            for item in data:
                samples.append({
                    "imgname": item.get("imgname", item.get("image", "")),
                    "query": item.get("query", item.get("question", "")),
                    "label": str(item.get("label", item.get("answer", ""))),
                })

    random.seed(42)
    subset = random.sample(samples, min(n, len(samples)))

    counts = {"correct": 0, "number_hallucination": 0, "wrong_other": 0}
    t0 = time.time()

    for i, sample in enumerate(subset):
        img_path = os.path.join(test_dir, "png", sample["imgname"])
        if not os.path.exists(img_path):
            continue

        try:
            pv = load_image_single_tile(img_path)
            pred = run_chat(model, tokenizer, pv,
                           f"{sample['query']}\nAnswer concisely.", max_new_tokens=64)
        except Exception:
            pred = ""

        gt = sample["label"]
        correct = _relaxed_match(pred, gt)
        h_type = _classify_hallucination(pred, gt, correct)
        counts[h_type] += 1

        if (i + 1) % 50 == 0:
            acc = counts["correct"] / (i + 1)
            print(f"  [{i+1}/{n}] Acc: {acc:.4f}")

    total = sum(counts.values())
    result = {
        "total": total,
        "correct": counts["correct"],
        "number_hallucination": counts["number_hallucination"],
        "wrong_other": counts["wrong_other"],
        "accuracy": counts["correct"] / max(total, 1),
        "halluc_rate": counts["number_hallucination"] / max(total, 1),
    }

    print(f"\n  Accuracy: {result['accuracy']:.4f}")
    print(f"  Halluc Rate: {result['halluc_rate']:.4f}")
    print(f"  Time: {time.time()-t0:.0f}s")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/NetDisk/j_son/internvl_35")
    parser.add_argument("--sfa_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_dir", default="/NetDisk/juyeon/train/chartQA/ChartQA Dataset/test")
    parser.add_argument("--label", default="SFA Ablation")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_sfa_model(args.model_path, args.sfa_checkpoint, device=args.device)

    # 1. ChartQA eval
    chartqa_results = run_chartqa_eval(model, tokenizer, args.test_dir)
    with open(os.path.join(args.output_dir, "chartqa_eval.json"), "w") as f:
        json.dump(chartqa_results, f, indent=2, ensure_ascii=False)

    # 2. Hallucination subset
    halluc_results = run_hallucination_subset(model, tokenizer, args.test_dir)
    with open(os.path.join(args.output_dir, "hallucination.json"), "w") as f:
        json.dump(halluc_results, f, indent=2)

    # 3. Summary
    summary = {
        "label": args.label,
        "checkpoint": args.sfa_checkpoint,
        "chartqa_accuracy": chartqa_results["accuracy"],
        "chartqa_halluc_rate": chartqa_results["halluc_rate"],
        "halluc_200_accuracy": halluc_results["accuracy"],
        "halluc_200_rate": halluc_results["halluc_rate"],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 50)
    print(f"EVALUATION COMPLETE — {args.label}")
    print("=" * 50)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
