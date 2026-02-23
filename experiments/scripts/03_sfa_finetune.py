"""
SFA Fine-tuning: InternVL3.5 + SFA on ChartQA
================================================
Vision encoder (SFA-patched) + projector만 학습, LLM frozen.
LLM 4-bit 양자화 + gradient checkpointing으로 A100-40GB 단일 GPU에서 학습 가능.

Usage:
    conda activate docmllm
    cd /NetDisk/juyeon/DSA

    # Quick test (10 steps)
    python experiments/scripts/03_sfa_finetune.py \
        --mode test \
        --model_path /NetDisk/j_son/internvl_35

    # Full training (single GPU, ~8GB model + quantized LLM)
    python experiments/scripts/03_sfa_finetune.py \
        --mode train \
        --model_path /NetDisk/j_son/internvl_35 \
        --train_data "/NetDisk/juyeon/train/chartQA/ChartQA Dataset/train" \
        --output_dir experiments/results/03_sfa_train \
        --epochs 3 --lr 2e-5 --batch_size 1 --grad_accum 32

    # Evaluation (SFA model on ChartQA test)
    python experiments/scripts/03_sfa_finetune.py \
        --mode eval \
        --model_path /NetDisk/j_son/internvl_35 \
        --sfa_checkpoint experiments/results/03_sfa_train/best.pth \
        --output_dir experiments/results/03_sfa_eval
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_internvl, load_image_single_tile, run_chat
from importlib import import_module
_sfa_int = import_module("03_sfa_integration")
patch_internvit_with_sfa = _sfa_int.patch_internvit_with_sfa


# ─── Dataset ────────────────────────────────────────────

class ChartQADataset(Dataset):
    """ChartQA train/test dataset for VLM fine-tuning."""

    def __init__(self, data_dir, split="train", max_samples=None, input_size=448):
        self.data_dir = data_dir
        self.input_size = input_size
        self.samples = []

        for fname in [f"{split}_human.json", f"{split}_augmented.json"]:
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                fpath = os.path.join(data_dir, "annotations", fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    data = json.load(f)
                for item in data:
                    self.samples.append({
                        "imgname": item.get("imgname", item.get("image", "")),
                        "query": item.get("query", item.get("question", "")),
                        "label": str(item.get("label", item.get("answer", ""))),
                    })

        if max_samples and len(self.samples) > max_samples:
            random.seed(42)
            self.samples = random.sample(self.samples, max_samples)

        print(f"[ChartQA {split}] {len(self.samples)} samples")

        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = os.path.join(self.data_dir, "png", item["imgname"])

        from PIL import Image
        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.transform(image)
        except Exception:
            pixel_values = torch.zeros(3, self.input_size, self.input_size)

        return {
            "pixel_values": pixel_values,
            "question": item["query"],
            "answer": item["label"],
        }


# ─── DDP Helpers ─────────────────────────────────────────

def _setup_ddp():
    """DDP 초기화. torchrun이 환경변수를 세팅해줌."""
    if "RANK" not in os.environ:
        return 0, 0, 1  # rank, local_rank, world_size (single GPU)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_main(rank):
    return rank == 0


# ─── Helpers ─────────────────────────────────────────────

def _resolve_img_context_token_id(model, tokenizer):
    """model.img_context_token_id를 확인하고 없으면 tokenizer에서 찾아 설정."""
    img_ctx_id = getattr(model, "img_context_token_id", None)
    if img_ctx_id is None:
        for name in ["<IMG_CONTEXT>", "<image>", "<img>"]:
            tid = tokenizer.convert_tokens_to_ids(name)
            if tid != tokenizer.unk_token_id:
                img_ctx_id = tid
                break
        if img_ctx_id is None:
            img_ctx_id = 151671  # InternVL3.5 default
        model.img_context_token_id = img_ctx_id
    return img_ctx_id


# ─── Loss computation ───────────────────────────────────

def build_chat_input(tokenizer, question, answer, img_context_token_id, num_image_token):
    """
    InternVL3.5 chat 형식의 input_ids + labels 생성.
    Answer 부분에만 loss를 적용합니다 (question/image 부분은 -100으로 마스킹).
    """
    IGNORE_TOKEN_ID = -100

    # Build conversation parts
    # InternVL format: <img><IMG_CONTEXT>*N</img>\nquestion → answer
    img_tokens = [img_context_token_id] * num_image_token

    # Tokenize question part (no loss)
    q_text = f"{question}\nAnswer concisely."
    q_ids = tokenizer.encode(q_text, add_special_tokens=False)

    # Tokenize answer part (with loss)
    a_ids = tokenizer.encode(answer, add_special_tokens=False)

    # EOS token
    eos_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("</s>")

    # Assemble: [BOS] + img_tokens + q_ids + a_ids + [EOS]
    bos_id = tokenizer.bos_token_id
    input_ids = []
    labels = []

    if bos_id is not None:
        input_ids.append(bos_id)
        labels.append(IGNORE_TOKEN_ID)

    # Image tokens (no loss)
    input_ids.extend(img_tokens)
    labels.extend([IGNORE_TOKEN_ID] * num_image_token)

    # Question (no loss)
    input_ids.extend(q_ids)
    labels.extend([IGNORE_TOKEN_ID] * len(q_ids))

    # Answer (with loss)
    input_ids.extend(a_ids)
    labels.extend(a_ids)

    # EOS (with loss)
    if eos_id is not None:
        input_ids.append(eos_id)
        labels.append(eos_id)

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_fn_factory(tokenizer, img_context_token_id, num_image_token):
    """Dynamic collate with padding."""
    IGNORE_TOKEN_ID = -100

    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        all_input_ids = []
        all_labels = []

        for b in batch:
            ids, lbl = build_chat_input(
                tokenizer, b["question"], b["answer"],
                img_context_token_id, num_image_token,
            )
            all_input_ids.append(ids)
            all_labels.append(lbl)

        # Pad to max length in batch
        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_id = tokenizer.pad_token_id or 0

        padded_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        padded_labels = torch.full((len(batch), max_len), IGNORE_TOKEN_ID, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

        for i, (ids, lbl) in enumerate(zip(all_input_ids, all_labels)):
            padded_ids[i, :ids.shape[0]] = ids
            padded_labels[i, :lbl.shape[0]] = lbl
            attention_mask[i, :ids.shape[0]] = 1

        return {
            "pixel_values": pixel_values,
            "input_ids": padded_ids,
            "labels": padded_labels,
            "attention_mask": attention_mask,
        }

    return collate_fn


def compute_vl_loss(model, batch, device, dtype, llm_device=None):
    """
    Vision-Language loss 계산.

    Args:
        model: InternVL 모델
        batch: collated batch dict
        device: vision encoder 디바이스 (cuda:0)
        dtype: compute dtype
        llm_device: LLM 디바이스 (split GPU 시 cuda:1, 아니면 None=device와 동일)
    """
    if llm_device is None:
        llm_device = device

    pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
    input_ids = batch["input_ids"].to(llm_device)
    labels = batch["labels"].to(llm_device)
    attention_mask = batch["attention_mask"].to(llm_device)

    raw = model.module if hasattr(model, "module") else model

    # Vision encoder forward (on vision device)
    vit_embeds = raw.extract_feature(pixel_values)  # (B, 256, C) on device

    # LLM embedding (on llm_device)
    input_embeds = raw.language_model.get_input_embeddings()(input_ids).clone()

    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)
    flat_ids = input_ids.reshape(B * N)

    img_ctx_id = getattr(raw, "img_context_token_id", 151671)
    selected = (flat_ids == img_ctx_id)
    n_selected = selected.sum().item()

    # Move vit_embeds to LLM device if split
    vit_flat = vit_embeds.reshape(-1, C).to(llm_device)

    if n_selected > 0 and vit_flat.shape[0] > 0:
        n_use = min(n_selected, vit_flat.shape[0])
        sel_indices = selected.nonzero(as_tuple=True)[0][:n_use]
        input_embeds[sel_indices] = input_embeds[sel_indices] * 0.0 + vit_flat[:n_use].to(input_embeds.dtype)

    input_embeds = input_embeds.reshape(B, N, C)

    outputs = raw.language_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        return_dict=True,
    )

    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


# ─── Training ───────────────────────────────────────────

def _enable_gradient_checkpointing(model):
    """Vision encoder에 gradient checkpointing 활성화 (활성화 메모리 절감)."""
    vision_model = getattr(model, "vision_model", None)
    if vision_model is not None and hasattr(vision_model, "encoder"):
        vision_model.encoder.gradient_checkpointing = True
        # Wrap each encoder layer for checkpointing
        for layer in vision_model.encoder.layers:
            layer._orig_forward = layer.forward
            def _make_ckpt_forward(mod):
                def ckpt_forward(*args, **kwargs):
                    if mod.training:
                        return torch.utils.checkpoint.checkpoint(
                            mod._orig_forward, *args, use_reentrant=False, **kwargs
                        )
                    return mod._orig_forward(*args, **kwargs)
                return ckpt_forward
            layer.forward = _make_ckpt_forward(layer)
        print("  Gradient checkpointing enabled for vision encoder")


def train_sfa(args):
    """SFA fine-tuning main loop. 2-GPU model parallel 또는 single GPU 4-bit 양자화."""
    rank, local_rank, world_size = _setup_ddp()
    is_ddp = world_size > 1
    dtype = torch.bfloat16

    # GPU 수에 따라 로딩 전략 결정
    num_gpus = torch.cuda.device_count()
    use_split = (num_gpus >= 2) and (not is_ddp)

    if use_split:
        device = "cuda:0"       # vision encoder device
        llm_device = "cuda:1"   # LLM device
    else:
        device = f"cuda:{local_rank}" if is_ddp else "cuda"
        llm_device = device

    if _is_main(rank):
        print("=" * 60)
        if use_split:
            print(f"SFA Fine-tuning (2-GPU Model Parallel: vision→GPU0, LLM→GPU1)")
        elif is_ddp:
            print(f"SFA Fine-tuning (DDP: {world_size} GPUs)")
        else:
            print(f"SFA Fine-tuning (single GPU, quantized)")
        print("=" * 60)

    # 1. Load model
    model, tokenizer = load_internvl(
        args.model_path, device=device, dtype=dtype,
        quantize_llm=(not use_split and not is_ddp),
        split_gpu=use_split,
    )

    # 2. Patch with SFA
    model = patch_internvit_with_sfa(model)

    # Enable gradient checkpointing for vision encoder (only when backbone is trainable)
    if not args.freeze_backbone:
        _enable_gradient_checkpointing(model)
    else:
        # Explicitly disable any built-in gradient checkpointing
        vision_model = getattr(model, "vision_model", None)
        if vision_model is not None and hasattr(vision_model, "encoder"):
            vision_model.encoder.gradient_checkpointing = False
        if _is_main(rank):
            print("  Gradient checkpointing disabled (backbone frozen)")

    model.train()

    # Ensure LLM is in eval mode (frozen)
    if hasattr(model, "language_model"):
        model.language_model.eval()
        for p in model.language_model.parameters():
            p.requires_grad = False

    # Optionally freeze vision encoder backbone (SFA-only training)
    if args.freeze_backbone:
        frozen_count = 0
        sfa_count = 0
        for name, param in model.named_parameters():
            if "vision_model" in name:
                if "structural_bias" in name:
                    param.requires_grad = True
                    sfa_count += 1
                else:
                    param.requires_grad = False
                    frozen_count += 1
        if args.freeze_projector and hasattr(model, "mlp1"):
            for p in model.mlp1.parameters():
                p.requires_grad = False
        # Enable requires_grad on vision encoder embedding output so gradient
        # can flow through the encoder layers to SFA structural bias params.
        # Without this, gradient checkpointing or encoder internals may break
        # the computation graph when all backbone params are frozen.
        vision_model = getattr(model, "vision_model", None)
        if vision_model is not None:
            def _enable_grad_hook(module, args, output):
                if isinstance(output, torch.Tensor):
                    return output.requires_grad_(True)
                return output
            vision_model.embeddings.register_forward_hook(_enable_grad_hook)

        if _is_main(rank):
            print(f"  [freeze_backbone] Frozen {frozen_count} vision backbone params, kept {sfa_count} SFA params trainable")
            if args.freeze_projector:
                print(f"  [freeze_projector] Projector (mlp1) also frozen")

    # Get image token info
    img_ctx_id = _resolve_img_context_token_id(model, tokenizer)
    num_img_token = getattr(model, "num_image_token", 256)

    if _is_main(rank):
        print(f"  img_context_token_id: {img_ctx_id}")
        print(f"  num_image_token: {num_img_token}")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
        if use_split:
            mem0 = torch.cuda.memory_allocated("cuda:0") / 1024**3
            mem1 = torch.cuda.memory_allocated("cuda:1") / 1024**3
            print(f"  GPU 0 (vision): {mem0:.1f} GB | GPU 1 (LLM): {mem1:.1f} GB")
        else:
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            print(f"  GPU memory after setup: {allocated:.1f} GB")

    # 3. Wrap with DDP (only trainable params participate in gradient sync)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        raw_model = model.module
    else:
        raw_model = model

    # 4. Dataset
    train_dataset = ChartQADataset(args.train_data, split="train", max_samples=args.max_samples)
    collate_fn = collate_fn_factory(tokenizer, img_ctx_id, num_img_token)

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=2, pin_memory=True, collate_fn=collate_fn, drop_last=True,
    )

    # 5. Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95),
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = min(100, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 6. Training loop
    if _is_main(rank):
        os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "train_log.json")
    logs = []
    best_loss = float("inf")
    global_step = 0
    oom_count = 0

    if _is_main(rank):
        eff_bs = args.batch_size * args.grad_accum * world_size
        print(f"\n  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size} × grad_accum {args.grad_accum} × {world_size} GPUs = {eff_bs}")
        print(f"  Steps/epoch: {len(train_loader)}")
        print(f"  Total optimizer steps: {total_steps}")
        print(f"  LR: {args.lr}")
        print(f"  Mode: {'2-GPU split' if use_split else 'quantized' if not is_ddp else 'DDP'}")
        print()

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        if hasattr(raw_model, "language_model"):
            raw_model.language_model.eval()

        epoch_loss = 0
        epoch_steps = 0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            try:
                with torch.amp.autocast("cuda", dtype=dtype):
                    loss = compute_vl_loss(
                        raw_model if not is_ddp else model, batch,
                        device, dtype, llm_device=llm_device,
                    )
                    loss = loss / args.grad_accum
                loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    oom_count += 1
                    if _is_main(rank):
                        print(f"  [OOM] step {step}, skipping (total OOM: {oom_count})")
                    optimizer.zero_grad()
                    if oom_count > 10:
                        print("  [FATAL] Too many OOM errors. Reduce batch_size or enable more aggressive quantization.")
                        break
                    continue
                raise

            epoch_loss += loss.item() * args.grad_accum
            epoch_steps += 1

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if _is_main(rank) and (step + 1) % (args.grad_accum * 10) == 0:
                avg = epoch_loss / epoch_steps
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                mem0 = torch.cuda.memory_allocated("cuda:0") / 1024**3
                if use_split:
                    mem1 = torch.cuda.memory_allocated("cuda:1") / 1024**3
                    mem_str = f"G0:{mem0:.1f}GB G1:{mem1:.1f}GB"
                else:
                    mem_str = f"mem:{mem0:.1f}GB"
                print(f"  [Epoch {epoch+1}] step {step+1}/{len(train_loader)} | "
                      f"loss: {avg:.4f} | lr: {lr_now:.2e} | {mem_str} | {elapsed:.0f}s")

        # Epoch summary
        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0

        if _is_main(rank):
            print(f"\n  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} ({elapsed:.0f}s)")

            log_entry = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "lr": scheduler.get_last_lr()[0],
                "time_sec": elapsed,
                "oom_count": oom_count,
            }
            logs.append(log_entry)

            # Save checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_sfa_weights(raw_model, os.path.join(args.output_dir, "best.pth"))
                print(f"  Saved best checkpoint (loss={best_loss:.4f})")

            save_sfa_weights(raw_model, os.path.join(args.output_dir, "latest.pth"))

            with open(log_file, "w") as f:
                json.dump({"config": vars(args), "logs": logs}, f, indent=2)

        if is_ddp:
            dist.barrier()

    if _is_main(rank):
        print(f"\nTraining complete. Best loss: {best_loss:.4f}")
        print(f"Checkpoints saved to {args.output_dir}/")

    _cleanup_ddp()
    return raw_model


def save_sfa_weights(model, path):
    """SFA structural bias weights + vision encoder + projector 저장."""
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = param.data.cpu()
    torch.save(state, path)


def load_sfa_weights(model, path):
    """저장된 SFA weights 로드."""
    state = torch.load(path, map_location="cpu")
    missing = []
    for name, param in model.named_parameters():
        if name in state:
            param.data.copy_(state[name].to(param.dtype))
        elif param.requires_grad:
            missing.append(name)
    if missing:
        print(f"  [WARN] Missing {len(missing)} trainable params in checkpoint")
    print(f"  Loaded {len(state)} params from {path}")
    return model


# ─── Evaluation ─────────────────────────────────────────

def eval_sfa(args):
    """SFA 모델 ChartQA test 평가."""
    device = "cuda"
    dtype = torch.bfloat16

    model, tokenizer = load_internvl(args.model_path, device=device, dtype=dtype)
    model = patch_internvit_with_sfa(model)

    if args.sfa_checkpoint and os.path.exists(args.sfa_checkpoint):
        load_sfa_weights(model, args.sfa_checkpoint)
    model.eval()

    # Load test data
    test_dir = args.test_data or "/NetDisk/juyeon/train/chartQA/ChartQA Dataset/test"
    test_dataset = ChartQADataset(test_dir, split="test", max_samples=args.max_samples)

    results = {"total": 0, "correct": 0, "details": []}
    t0 = time.time()

    for i, sample in enumerate(test_dataset.samples):
        img_path = os.path.join(test_dir, "png", sample["imgname"])
        if not os.path.exists(img_path):
            continue

        try:
            pv = load_image_single_tile(img_path)
            pred = run_chat(model, tokenizer, pv, f"{sample['query']}\nAnswer concisely.", max_new_tokens=64)
        except Exception:
            pred = ""

        gt = sample["label"]
        correct = _relaxed_match(pred, gt)
        results["total"] += 1
        if correct:
            results["correct"] += 1
        results["details"].append({
            "question": sample["query"], "gt": gt, "pred": pred, "correct": correct
        })

        if (i + 1) % 50 == 0:
            acc = results["correct"] / results["total"]
            print(f"  [{i+1}/{len(test_dataset)}] Acc: {acc:.4f}")

    acc = results["correct"] / max(results["total"], 1)
    elapsed = time.time() - t0
    results["accuracy"] = acc

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"SFA Evaluation: ChartQA Relaxed Accuracy = {acc:.4f}")
    print(f"  Samples: {results['total']} | Time: {elapsed:.1f}s")
    print(f"  Saved to {args.output_dir}/eval_results.json")


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


# ─── Test mode ──────────────────────────────────────────

def test_training(args):
    """Quick training test — 10 steps만 실행하여 전체 파이프라인 동작 확인."""
    num_gpus = torch.cuda.device_count()
    use_split = num_gpus >= 2

    print("=" * 60)
    print(f"SFA Training Pipeline Test (10 steps, {'2-GPU split' if use_split else 'quantized LLM'})")
    print("=" * 60)

    device = "cuda:0" if use_split else "cuda"
    llm_device = "cuda:1" if use_split else device
    dtype = torch.bfloat16

    model, tokenizer = load_internvl(
        args.model_path, device=device, dtype=dtype,
        quantize_llm=(not use_split), split_gpu=use_split,
    )
    model = patch_internvit_with_sfa(model)
    _enable_gradient_checkpointing(model)
    model.train()

    if hasattr(model, "language_model"):
        model.language_model.eval()

    img_ctx_id = _resolve_img_context_token_id(model, tokenizer)
    num_img_token = getattr(model, "num_image_token", 256)

    print(f"  img_context_token_id: {img_ctx_id}")
    print(f"  num_image_token: {num_img_token}")

    # Synthetic test batch
    B = 2
    pixel_values = torch.randn(B, 3, 448, 448, device=device, dtype=dtype)

    # Build inputs
    questions = ["What is the value for Japan?", "How many bars are shown?"]
    answers = ["25.5", "5"]

    collate = collate_fn_factory(tokenizer, img_ctx_id, num_img_token)
    batch_items = [
        {"pixel_values": pixel_values[i].cpu().float(), "question": questions[i], "answer": answers[i]}
        for i in range(B)
    ]
    batch = collate(batch_items)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5
    )

    print("\nRunning 10 training steps...")
    for step in range(10):
        optimizer.zero_grad()
        try:
            loss = compute_vl_loss(model, batch, device, dtype, llm_device=llm_device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            print(f"  Step {step+1}: loss = {loss.item():.4f}")
        except Exception as e:
            print(f"  Step {step+1}: ERROR — {e}")
            import traceback; traceback.print_exc()
            return False

    print("\nTraining pipeline test PASSED")

    # Quick check: structural bias has been updated
    layer0 = model.vision_model.encoder.layers[0].attn
    if hasattr(layer0, "structural_bias"):
        w_row = layer0.structural_bias.w_row.data
        print(f"  Layer 0 w_row (should be non-zero after training): {w_row[:4].tolist()}")

    return True


# ─── Main ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFA Fine-tuning for InternVL3.5")
    parser.add_argument("--mode", choices=["test", "train", "eval"], required=True)
    parser.add_argument("--model_path", default="/NetDisk/j_son/internvl_35")
    parser.add_argument("--train_data", default="/NetDisk/juyeon/train/chartQA/ChartQA Dataset/train")
    parser.add_argument("--test_data", default=None)
    parser.add_argument("--sfa_checkpoint", default=None)
    parser.add_argument("--output_dir", default="experiments/results/03_sfa_train")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze vision encoder backbone; only train SFA structural bias + projector")
    parser.add_argument("--freeze_projector", action="store_true",
                        help="Also freeze projector (mlp1). Use with --freeze_backbone for SFA-only training")
    args = parser.parse_args()

    if args.mode == "test":
        test_training(args)
    elif args.mode == "train":
        train_sfa(args)
    elif args.mode == "eval":
        eval_sfa(args)


if __name__ == "__main__":
    main()
