"""
SFA+ADAT Training: Density-Guided Block Assignment
====================================================
SFA structural bias + density estimator block assignment로 학습합니다.
Backbone frozen 전략 사용 (Phase 2 ablation에서 검증).

DensityEstimator가 각 이미지의 텍스트 밀도를 예측하고,
밀도 값을 16개 블록으로 양자화하여 SFA의 block_embed에 전달합니다.

Usage:
    cd /NetDisk/juyeon/DSA
    CUDA_VISIBLE_DEVICES=1 python -u experiments/scripts/13_sfa_adat_train.py \
        --output_dir experiments/results/07_sfa_adat \
        --device cuda:0
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
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_internvl
from importlib import import_module

_sfa_int = import_module("03_sfa_integration")
patch_internvit_with_sfa = _sfa_int.patch_internvit_with_sfa

_sfa_mod = import_module("02_sfa_module")
density_to_block_ids = _sfa_mod.density_to_block_ids
set_block_ids_on_model = _sfa_mod.set_block_ids_on_model

_density = import_module("01_density_estimator")
DensityEstimator = _density.DensityEstimator

# Reuse from 03_sfa_finetune
_finetune = import_module("03_sfa_finetune")
ChartQADataset = _finetune.ChartQADataset
collate_fn_factory = _finetune.collate_fn_factory
compute_vl_loss = _finetune.compute_vl_loss
save_sfa_weights = _finetune.save_sfa_weights


def load_density_estimator(checkpoint_path, device="cuda"):
    """Load pretrained density estimator (frozen)."""
    model = DensityEstimator()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded density estimator from {checkpoint_path}")
    return model


def compute_block_ids_batch(density_model, pixel_values, num_blocks=16):
    """
    Run density estimator on a batch and convert to block_ids.

    Args:
        density_model: DensityEstimator (frozen)
        pixel_values: [B, 3, 448, 448] tensor
        num_blocks: number of discrete blocks

    Returns:
        block_ids_list: list of [1024] LongTensors, one per image
    """
    with torch.no_grad():
        # Density estimator uses float32 (BatchNorm incompatible with bfloat16)
        density_maps = density_model(pixel_values.float())  # [B, 1, 28, 28]

    block_ids_list = []
    for i in range(density_maps.shape[0]):
        dm = density_maps[i]  # [1, 28, 28]
        bid = density_to_block_ids(dm, num_blocks=num_blocks, grid_h=32, grid_w=32)
        block_ids_list.append(bid)

    return block_ids_list


def _resolve_img_context_token_id(model, tokenizer):
    """model.img_context_token_id를 확인하고 없으면 설정."""
    img_ctx_id = getattr(model, "img_context_token_id", None)
    if img_ctx_id is None:
        for name in ["<IMG_CONTEXT>", "<image>", "<img>"]:
            tid = tokenizer.convert_tokens_to_ids(name)
            if tid != tokenizer.unk_token_id:
                img_ctx_id = tid
                break
        if img_ctx_id is None:
            img_ctx_id = 151671
        model.img_context_token_id = img_ctx_id
    return img_ctx_id


def train_sfa_adat(args):
    """SFA+ADAT fine-tuning: backbone frozen + density-guided block assignment."""
    dtype = torch.bfloat16
    device = args.device

    # Determine GPU strategy
    num_gpus = torch.cuda.device_count()
    use_split = num_gpus >= 2

    if use_split:
        vision_device = device
        llm_device = "cuda:1" if device == "cuda:0" else "cuda:0"
    else:
        vision_device = device
        llm_device = device

    print("=" * 60)
    print("SFA+ADAT Training (Density-Guided Block Assignment)")
    print("=" * 60)

    # 1. Load model
    model, tokenizer = load_internvl(
        args.model_path, device=vision_device, dtype=dtype,
        quantize_llm=(not use_split), split_gpu=use_split,
    )

    # 2. Patch with SFA
    model = patch_internvit_with_sfa(model)

    # 3. Load density estimator
    density_model = load_density_estimator(args.density_checkpoint, device=vision_device)

    # 4. Freeze backbone (SFA-only training strategy)
    # Disable gradient checkpointing (backbone frozen)
    vision_model = getattr(model, "vision_model", None)
    if vision_model is not None and hasattr(vision_model, "encoder"):
        vision_model.encoder.gradient_checkpointing = False
    print("  Gradient checkpointing disabled (backbone frozen)")

    model.train()
    if hasattr(model, "language_model"):
        model.language_model.eval()
        for p in model.language_model.parameters():
            p.requires_grad = False

    # Freeze backbone, keep SFA params trainable
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

    # Freeze projector
    if hasattr(model, "mlp1"):
        for p in model.mlp1.parameters():
            p.requires_grad = False

    # Enable requires_grad on embedding output for gradient flow
    if vision_model is not None:
        def _enable_grad_hook(module, args_tuple, output):
            if isinstance(output, torch.Tensor):
                return output.requires_grad_(True)
            return output
        vision_model.embeddings.register_forward_hook(_enable_grad_hook)

    print(f"  [freeze_backbone] Frozen {frozen_count} backbone params, kept {sfa_count} SFA params trainable")
    print(f"  [freeze_projector] Projector (mlp1) also frozen")

    img_ctx_id = _resolve_img_context_token_id(model, tokenizer)
    num_img_token = getattr(model, "num_image_token", 256)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  img_context_token_id: {img_ctx_id}")
    print(f"  num_image_token: {num_img_token}")
    print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.4f}%)")

    if use_split:
        mem0 = torch.cuda.memory_allocated(vision_device) / 1024**3
        mem1 = torch.cuda.memory_allocated(llm_device) / 1024**3
        print(f"  GPU 0 (vision): {mem0:.1f} GB | GPU 1 (LLM): {mem1:.1f} GB")

    # 5. Dataset
    train_dataset = ChartQADataset(args.train_data, split="train", max_samples=args.max_samples)
    collate_fn = collate_fn_factory(tokenizer, img_ctx_id, num_img_token)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
        collate_fn=collate_fn, drop_last=True,
    )

    # 6. Optimizer + scheduler
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

    # 7. Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "train_log.json")
    logs = []
    best_loss = float("inf")
    global_step = 0
    oom_count = 0

    eff_bs = args.batch_size * args.grad_accum
    print(f"\n  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x grad_accum {args.grad_accum} = {eff_bs}")
    print(f"  Steps/epoch: {len(train_loader)}")
    print(f"  Total optimizer steps: {total_steps}")
    print(f"  LR: {args.lr}")
    print(f"  Density blocks: {args.num_blocks}")
    print(f"  Mode: {'2-GPU split' if use_split else 'single GPU'}")
    print()

    for epoch in range(args.epochs):
        model.train()
        if hasattr(model, "language_model"):
            model.language_model.eval()

        epoch_loss = 0
        epoch_steps = 0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            try:
                pixel_values = batch["pixel_values"].to(device=vision_device, dtype=dtype)

                # Run density estimator → block_ids
                block_ids_list = compute_block_ids_batch(
                    density_model, pixel_values, num_blocks=args.num_blocks
                )
                # Use first image's block_ids (same structure for all in batch)
                # Note: ideally per-image block_ids, but SFA structural bias is shared
                # across the batch. Use batch[0] as representative.
                block_ids = block_ids_list[0].to(vision_device)
                set_block_ids_on_model(model, block_ids)

                with torch.amp.autocast("cuda", dtype=dtype):
                    loss = compute_vl_loss(
                        model, batch, vision_device, dtype, llm_device=llm_device,
                    )
                    loss = loss / args.grad_accum
                loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    oom_count += 1
                    print(f"  [OOM] step {step}, skipping (total OOM: {oom_count})")
                    optimizer.zero_grad()
                    if oom_count > 10:
                        print("  [FATAL] Too many OOM errors.")
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

            if (step + 1) % (args.grad_accum * 10) == 0:
                avg = epoch_loss / epoch_steps
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                mem0 = torch.cuda.memory_allocated(vision_device) / 1024**3
                if use_split:
                    mem1 = torch.cuda.memory_allocated(llm_device) / 1024**3
                    mem_str = f"G0:{mem0:.1f}GB G1:{mem1:.1f}GB"
                else:
                    mem_str = f"mem:{mem0:.1f}GB"
                print(f"  [Epoch {epoch+1}] step {step+1}/{len(train_loader)} | "
                      f"loss: {avg:.4f} | lr: {lr_now:.2e} | {mem_str} | {elapsed:.0f}s")

        # Epoch summary
        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} ({elapsed:.0f}s)")

        log_entry = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "lr": scheduler.get_last_lr()[0],
            "time_sec": elapsed,
            "oom_count": oom_count,
        }
        logs.append(log_entry)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_sfa_weights(model, os.path.join(args.output_dir, "best.pth"))
            print(f"  Saved best checkpoint (loss={best_loss:.4f})")

        save_sfa_weights(model, os.path.join(args.output_dir, "latest.pth"))

        with open(log_file, "w") as f:
            json.dump({"config": vars(args), "logs": logs}, f, indent=2)

    # Clear block_ids
    set_block_ids_on_model(model, None)

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="SFA+ADAT Training")
    parser.add_argument("--model_path", default="/NetDisk/j_son/internvl_35")
    parser.add_argument("--train_data", default="/NetDisk/juyeon/train/chartQA/ChartQA Dataset/train")
    parser.add_argument("--density_checkpoint", default="experiments/results/01_density/best.pth")
    parser.add_argument("--output_dir", default="experiments/results/07_sfa_adat")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_blocks", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    train_sfa_adat(args)


if __name__ == "__main__":
    main()
