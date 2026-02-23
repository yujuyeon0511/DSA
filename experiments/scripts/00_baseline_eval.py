"""
Experiment 0: InternVL3.5-8B Baseline Evaluation
=================================================
논문 Table 1의 baseline 수치를 생성합니다.
ChartQA, DocVQA, TextVQA, OCRBench, AI2D 5개 벤치마크 평가.

Usage:
    conda activate docmllm
    python experiments/scripts/00_baseline_eval.py \
        --model_path /NetDisk/j_son/Model_original/InternVL_35 \
        --output_dir experiments/results/00_baseline \
        --benchmarks chartqa docvqa textvqa ocrbench ai2d
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer


def load_model(model_path):
    """InternVL3.5-8B 모델 로드 (meta tensor issue 회피)"""
    from transformers import AutoModel, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True)

    # Load state dict manually to avoid meta tensor
    import json as _json
    from safetensors.torch import load_file as load_safetensors
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = _json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    state_dict = {}
    for shard_file in shard_files:
        shard_path = os.path.join(model_path, shard_file)
        print(f"  Loading {shard_file}...")
        shard_state = load_safetensors(shard_path)
        state_dict.update(shard_state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    del state_dict

    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    return model, tokenizer


def load_chartqa_test(data_root="/NetDisk/juyeon/train/chartQA/ChartQA Dataset/test"):
    """ChartQA test set 로드"""
    import glob
    samples = []
    for split_file in ["test_human.json", "test_augmented.json"]:
        fpath = os.path.join(data_root, "annotations", split_file)
        if not os.path.exists(fpath):
            # try alternative path
            fpath = os.path.join(data_root, split_file)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            for item in data:
                samples.append({
                    "question": item["query"] if "query" in item else item.get("question", ""),
                    "answer": str(item.get("label", item.get("answer", ""))),
                    "image_path": os.path.join(data_root, "png", item["imgname"] if "imgname" in item else item.get("image", "")),
                    "source": split_file.replace(".json", ""),
                })
    print(f"[ChartQA] Loaded {len(samples)} test samples")
    return samples


def load_docvqa_test(data_root="/NetDisk/juyeon/train/cauldron_data/docvqa"):
    """DocVQA from cauldron format"""
    samples = []
    jsonl_path = os.path.join(data_root, "output_train.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if i >= 500:  # eval subset
                    break
                item = json.loads(line)
                # cauldron format parsing
                samples.append({
                    "question": item.get("question", item.get("conversations", [{}])[0].get("value", "")),
                    "answer": item.get("answer", item.get("conversations", [{}])[-1].get("value", "")),
                    "image_path": os.path.join(data_root, "images", item.get("image", "")),
                    "source": "docvqa",
                })
    print(f"[DocVQA] Loaded {len(samples)} test samples")
    return samples


def relaxed_accuracy(pred: str, gt: str, tolerance=0.05) -> float:
    """ChartQA relaxed accuracy: 숫자는 ±5% tolerance"""
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


def anls_score(pred: str, gt: str) -> float:
    """Average Normalized Levenshtein Similarity"""
    pred = pred.strip().lower()
    gt = gt.strip().lower()
    if not gt:
        return 0.0
    # Levenshtein distance
    m, n = len(pred), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + (0 if pred[i-1] == gt[j-1] else 1)
            )
    dist = dp[m][n]
    nls = 1 - dist / max(m, n) if max(m, n) > 0 else 1.0
    return nls if nls >= 0.5 else 0.0  # threshold 0.5


def load_image_internvl(image_path, input_size=448, max_num=12):
    """InternVL 형식으로 이미지 전처리 (dynamic_preprocess)"""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    image = Image.open(image_path).convert("RGB")
    # Simple single-tile for speed (no dynamic_preprocess)
    pixel_values = transform(image).unsqueeze(0)
    return pixel_values


def run_inference(model, tokenizer, image_path, question, max_new_tokens=256):
    """단일 샘플 추론"""
    try:
        pixel_values = load_image_internvl(image_path)
        device = next(model.parameters()).device
        pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
    except Exception as e:
        print(f"  [WARN] Failed to load image {image_path}: {e}")
        return ""

    prompt = f"{question}\nAnswer the question concisely."
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)

    try:
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        return response.strip()
    except Exception as e:
        print(f"  [WARN] Inference failed: {e}")
        return ""


def evaluate_benchmark(model, tokenizer, samples, metric_fn, benchmark_name, output_dir):
    """벤치마크 평가 루프"""
    results = []
    scores = []
    start = time.time()

    for i, sample in enumerate(samples):
        pred = run_inference(model, tokenizer, sample["image_path"], sample["question"])
        score = metric_fn(pred, sample["answer"])
        scores.append(score)
        results.append({
            "idx": i,
            "question": sample["question"],
            "gt": sample["answer"],
            "pred": pred,
            "score": score,
            "source": sample.get("source", ""),
        })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            avg = sum(scores) / len(scores)
            print(f"  [{benchmark_name}] {i+1}/{len(samples)} | Avg: {avg:.4f} | {elapsed:.1f}s")

    avg_score = sum(scores) / len(scores) if scores else 0.0
    elapsed = time.time() - start

    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{benchmark_name}_results.json"), "w") as f:
        json.dump({
            "benchmark": benchmark_name,
            "num_samples": len(samples),
            "avg_score": avg_score,
            "elapsed_sec": elapsed,
            "details": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"  [{benchmark_name}] Final: {avg_score:.4f} ({len(samples)} samples, {elapsed:.1f}s)")
    return avg_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/NetDisk/j_son/internvl_35")
    parser.add_argument("--output_dir", default="experiments/results/00_baseline")
    parser.add_argument("--benchmarks", nargs="+", default=["chartqa"])
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples per benchmark for quick test")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path)
    print("Model loaded.")

    summary = {}

    for bench in args.benchmarks:
        print(f"\n{'='*60}")
        print(f"Evaluating: {bench}")
        print(f"{'='*60}")

        if bench == "chartqa":
            samples = load_chartqa_test()
            if args.max_samples:
                samples = samples[:args.max_samples]
            score = evaluate_benchmark(model, tokenizer, samples, relaxed_accuracy, "chartqa", args.output_dir)
            summary["ChartQA (Relaxed Acc)"] = score

        elif bench == "docvqa":
            samples = load_docvqa_test()
            if args.max_samples:
                samples = samples[:args.max_samples]
            score = evaluate_benchmark(model, tokenizer, samples, anls_score, "docvqa", args.output_dir)
            summary["DocVQA (ANLS)"] = score

        else:
            print(f"  [SKIP] {bench} loader not implemented yet")

    # Save summary
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
