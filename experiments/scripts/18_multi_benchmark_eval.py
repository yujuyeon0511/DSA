"""
Multi-Benchmark Evaluation
===========================
Baseline vs SFA+ADAT+SCR 비교 평가를 DocVQA, InfographicVQA, DVQA, FigureQA, HiTab에서 실행.

Usage:
    # Sanity check (20 samples, DocVQA only)
    CUDA_VISIBLE_DEVICES=1 python -u experiments/scripts/18_multi_benchmark_eval.py \
        --mode both --benchmarks docvqa --max_samples 20 --device cuda:0

    # Full evaluation (500 samples x 5 benchmarks)
    CUDA_VISIBLE_DEVICES=1 python -u experiments/scripts/18_multi_benchmark_eval.py \
        --mode both \
        --benchmarks docvqa infographic_vqa dvqa figureqa hitab \
        --max_samples 500 --device cuda:0
"""

import argparse
import json
import os
import random
import sys
import time

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_internvl, load_image_single_tile, run_chat
from importlib import import_module

_sfa_int = import_module("03_sfa_integration")
patch_internvit_with_sfa = _sfa_int.patch_internvit_with_sfa

_sfa_mod = import_module("02_sfa_module")
density_to_block_ids = _sfa_mod.density_to_block_ids
set_block_ids_on_model = _sfa_mod.set_block_ids_on_model

_density = import_module("01_density_estimator")
DensityEstimator = _density.DensityEstimator


# ─── Cauldron data root ───
CAULDRON_ROOT = "/NetDisk/juyeon/train/cauldron_data"

# ─── Benchmark configs ───
BENCHMARK_CONFIG = {
    "docvqa": {
        "data_dir": os.path.join(CAULDRON_ROOT, "docvqa"),
        "metric": "anls",
        "max_new_tokens": 128,
    },
    "infographic_vqa": {
        "data_dir": os.path.join(CAULDRON_ROOT, "infographic_vqa"),
        "metric": "anls",
        "max_new_tokens": 128,
    },
    "dvqa": {
        "data_dir": os.path.join(CAULDRON_ROOT, "dvqa"),
        "metric": "exact_match",
        "max_new_tokens": 64,
    },
    "figureqa": {
        "data_dir": os.path.join(CAULDRON_ROOT, "figureqa"),
        "metric": "exact_match",
        "max_new_tokens": 16,
    },
    "hitab": {
        "data_dir": os.path.join(CAULDRON_ROOT, "hitab"),
        "metric": "anls",
        "max_new_tokens": 128,
    },
}


# ═══════════════════════════════════════════
# Metric Functions
# ═══════════════════════════════════════════

def anls_score(pred: str, gt: str) -> float:
    """Average Normalized Levenshtein Similarity (DocVQA standard)."""
    pred = pred.strip().lower()
    gt = gt.strip().lower()
    if not gt:
        return 0.0
    m, n = len(pred), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (0 if pred[i - 1] == gt[j - 1] else 1),
            )
    dist = dp[m][n]
    nls = 1 - dist / max(m, n) if max(m, n) > 0 else 1.0
    return nls if nls >= 0.5 else 0.0


def relaxed_accuracy(pred: str, gt: str, tolerance=0.05) -> float:
    """ChartQA-style relaxed accuracy: exact match or numeric within 5%."""
    pred = pred.strip().lower()
    gt = gt.strip().lower()
    if pred == gt:
        return 1.0
    try:
        pred_num = float(pred.replace(",", "").replace("%", ""))
        gt_num = float(gt.replace(",", "").replace("%", ""))
        if gt_num == 0:
            return 1.0 if pred_num == 0 else 0.0
        if abs(pred_num - gt_num) / abs(gt_num) <= tolerance:
            return 1.0
    except ValueError:
        pass
    return 0.0


def exact_match(pred: str, gt: str) -> float:
    """Case-insensitive exact match (for yes/no answers)."""
    pred_norm = pred.strip().lower().rstrip(".")
    gt_norm = gt.strip().lower().rstrip(".")
    return 1.0 if pred_norm == gt_norm else 0.0


METRIC_FNS = {
    "anls": anls_score,
    "relaxed_accuracy": relaxed_accuracy,
    "exact_match": exact_match,
}


# ═══════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════

def load_cauldron_benchmark(benchmark_name, max_samples=500, seed=42):
    """Load samples from a cauldron-format JSONL dataset."""
    config = BENCHMARK_CONFIG[benchmark_name]
    data_dir = config["data_dir"]
    jsonl_path = os.path.join(data_dir, "output_train.jsonl")

    if not os.path.exists(jsonl_path):
        print(f"  [WARN] {jsonl_path} not found, skipping {benchmark_name}")
        return []

    all_samples = []
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line)
            img_rel = item.get("image", "")
            all_samples.append({
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "image_path": os.path.join(data_dir, img_rel),
            })

    if max_samples and len(all_samples) > max_samples:
        random.seed(seed)
        all_samples = random.sample(all_samples, max_samples)

    print(f"  [{benchmark_name}] Loaded {len(all_samples)} samples")
    return all_samples


# ═══════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════

def load_baseline_model(model_path, device="cuda"):
    """Load vanilla InternVL3.5 (no SFA)."""
    model, tokenizer = load_internvl(model_path, device=device, dtype=torch.bfloat16)
    model.eval()
    return model, tokenizer, None  # no density_model


def load_sfa_model(model_path, sfa_checkpoint, density_checkpoint, device="cuda"):
    """Load InternVL3.5 with SFA + density estimator."""
    dtype = torch.bfloat16
    model, tokenizer = load_internvl(model_path, device=device, dtype=dtype)
    model = patch_internvit_with_sfa(model, num_patches_h=32, num_patches_w=32)

    if sfa_checkpoint and os.path.isfile(sfa_checkpoint):
        print(f"Loading SFA checkpoint: {sfa_checkpoint}")
        ckpt = torch.load(sfa_checkpoint, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"  Loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    model.eval()

    # Load density estimator
    density_model = DensityEstimator()
    d_ckpt = torch.load(density_checkpoint, map_location="cpu", weights_only=True)
    density_model.load_state_dict(d_ckpt)
    density_model = density_model.to(device).eval()
    for p in density_model.parameters():
        p.requires_grad = False
    print(f"Loaded density estimator from {density_checkpoint}")

    return model, tokenizer, density_model


# ═══════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════

def evaluate_benchmark(model, tokenizer, density_model, samples, benchmark_name,
                       device, num_blocks=16, use_sfa=True):
    """Evaluate one benchmark. Returns results dict."""
    config = BENCHMARK_CONFIG[benchmark_name]
    metric_fn = METRIC_FNS[config["metric"]]
    max_new_tokens = config["max_new_tokens"]

    density_transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    scores = []
    details = []
    skipped = 0
    t0 = time.time()

    for i, sample in enumerate(samples):
        if not os.path.exists(sample["image_path"]):
            skipped += 1
            continue

        try:
            if use_sfa and density_model is not None:
                img = Image.open(sample["image_path"]).convert("RGB")
                density_input = density_transform(img).unsqueeze(0)
                with torch.no_grad():
                    density_map = density_model(density_input.to(device))
                block_ids = density_to_block_ids(density_map[0], num_blocks=num_blocks)
                set_block_ids_on_model(model, block_ids.to(device))

            pv = load_image_single_tile(sample["image_path"])
            pred = run_chat(model, tokenizer, pv, sample["question"],
                            max_new_tokens=max_new_tokens, device=device)
        except Exception as e:
            pred = ""

        score = metric_fn(pred, sample["answer"])
        scores.append(score)
        details.append({
            "idx": i,
            "question": sample["question"][:200],
            "gt": sample["answer"],
            "pred": pred,
            "score": score,
        })

        if (i + 1) % 100 == 0:
            avg = sum(scores) / len(scores)
            elapsed = time.time() - t0
            print(f"  [{benchmark_name}] {i+1}/{len(samples)} | "
                  f"Avg {config['metric']}: {avg:.4f} | {elapsed:.0f}s")

    # Clear block_ids
    if use_sfa:
        set_block_ids_on_model(model, None)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    elapsed = time.time() - t0

    mode_str = "SFA+ADAT+SCR" if use_sfa else "Baseline"
    print(f"  [{benchmark_name}] {mode_str} Final: {avg_score:.4f} "
          f"({len(scores)} samples, {skipped} skipped, {elapsed:.0f}s)")

    return {
        "benchmark": benchmark_name,
        "mode": mode_str.lower().replace("+", "_"),
        "metric": config["metric"],
        "num_samples": len(scores),
        "skipped": skipped,
        "avg_score": avg_score,
        "elapsed_sec": round(elapsed, 1),
        "details": details,
    }


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def print_comparison_table(all_results, benchmarks):
    """Print comparison table to console."""
    print("\n" + "=" * 70)
    print("MULTI-BENCHMARK COMPARISON")
    print("=" * 70)
    print(f"{'Benchmark':<20} | {'Metric':<16} | {'Baseline':>8} | {'SFA+ADAT+SCR':>12} | {'Delta':>8}")
    print("-" * 70)

    for bench in benchmarks:
        bkey = f"baseline_{bench}"
        skey = f"sfa_{bench}"
        if bkey in all_results and skey in all_results:
            metric_name = all_results[bkey]["metric"]
            b_score = all_results[bkey]["avg_score"]
            s_score = all_results[skey]["avg_score"]
            delta = s_score - b_score
            sign = "+" if delta >= 0 else ""
            print(f"{bench:<20} | {metric_name:<16} | {b_score:>8.4f} | {s_score:>12.4f} | {sign}{delta:>7.4f}")
        elif bkey in all_results:
            metric_name = all_results[bkey]["metric"]
            b_score = all_results[bkey]["avg_score"]
            print(f"{bench:<20} | {metric_name:<16} | {b_score:>8.4f} | {'--':>12} | {'--':>8}")
        elif skey in all_results:
            metric_name = all_results[skey]["metric"]
            s_score = all_results[skey]["avg_score"]
            print(f"{bench:<20} | {metric_name:<16} | {'--':>8} | {s_score:>12.4f} | {'--':>8}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Multi-benchmark evaluation")
    parser.add_argument("--model_path", default="/NetDisk/j_son/internvl_35")
    parser.add_argument("--sfa_checkpoint", default="experiments/results/08_scr/best.pth")
    parser.add_argument("--density_checkpoint", default="experiments/results/01_density/best.pth")
    parser.add_argument("--output_dir", default="experiments/results/09_multi_benchmark")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["docvqa", "infographic_vqa", "dvqa", "figureqa", "hitab"])
    parser.add_argument("--mode", choices=["baseline", "sfa", "both"], default="both")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--num_blocks", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Pre-load all benchmark data (CPU, fast)
    print("Loading benchmark data...")
    benchmark_data = {}
    for bench_name in args.benchmarks:
        if bench_name not in BENCHMARK_CONFIG:
            print(f"  [WARN] Unknown benchmark: {bench_name}, skipping")
            continue
        benchmark_data[bench_name] = load_cauldron_benchmark(bench_name, args.max_samples)

    all_results = {}

    # ─── BASELINE ───
    if args.mode in ("baseline", "both"):
        print("\n" + "=" * 50)
        print("LOADING BASELINE MODEL")
        print("=" * 50)
        model, tokenizer, _ = load_baseline_model(args.model_path, args.device)

        for bench_name in args.benchmarks:
            if bench_name not in benchmark_data or not benchmark_data[bench_name]:
                continue
            print(f"\n--- Evaluating Baseline on {bench_name} ---")
            results = evaluate_benchmark(
                model, tokenizer, None,
                benchmark_data[bench_name], bench_name,
                args.device, use_sfa=False,
            )
            key = f"baseline_{bench_name}"
            all_results[key] = results

            out_path = os.path.join(args.output_dir, f"{key}.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  Saved → {out_path}")

        del model, tokenizer
        torch.cuda.empty_cache()
        print("\nBaseline model unloaded.")

    # ─── SFA+ADAT+SCR ───
    if args.mode in ("sfa", "both"):
        print("\n" + "=" * 50)
        print("LOADING SFA+ADAT+SCR MODEL")
        print("=" * 50)
        model, tokenizer, density_model = load_sfa_model(
            args.model_path, args.sfa_checkpoint, args.density_checkpoint, args.device,
        )

        for bench_name in args.benchmarks:
            if bench_name not in benchmark_data or not benchmark_data[bench_name]:
                continue
            print(f"\n--- Evaluating SFA+ADAT+SCR on {bench_name} ---")
            results = evaluate_benchmark(
                model, tokenizer, density_model,
                benchmark_data[bench_name], bench_name,
                args.device, num_blocks=args.num_blocks, use_sfa=True,
            )
            key = f"sfa_{bench_name}"
            all_results[key] = results

            out_path = os.path.join(args.output_dir, f"{key}.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  Saved → {out_path}")

        del model, tokenizer, density_model
        torch.cuda.empty_cache()
        print("\nSFA model unloaded.")

    # ─── Summary ───
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_samples": args.max_samples,
        "sfa_checkpoint": args.sfa_checkpoint,
        "density_checkpoint": args.density_checkpoint,
        "results": {},
    }
    for bench in args.benchmarks:
        entry = {}
        bkey = f"baseline_{bench}"
        skey = f"sfa_{bench}"
        if bkey in all_results:
            entry["baseline"] = all_results[bkey]["avg_score"]
        if skey in all_results:
            entry["sfa"] = all_results[skey]["avg_score"]
        if "baseline" in entry and "sfa" in entry:
            entry["delta"] = round(entry["sfa"] - entry["baseline"], 4)
        if entry:
            summary["results"][bench] = entry

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print_comparison_table(all_results, args.benchmarks)
    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
