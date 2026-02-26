"""
Mixed-Dataset SFA+SCR Training
================================
다양한 문서 데이터셋(ChartQA, DocVQA, InfographicVQA, DVQA, FigureQA)을
혼합하여 SFA structural bias를 학습합니다.

ChartQA-only 학습 대비 cross-benchmark transfer 개선이 목표.

L_total = L_task + alpha * L_entropy  (SCR)

Usage:
    cd /NetDisk/juyeon/DSA
    CUDA_VISIBLE_DEVICES=1 /home/juyeon/miniconda3/envs/docmllm/bin/python -u \
        experiments/scripts/19_mixed_train.py \
        --datasets chartqa docvqa infographic_vqa dvqa figureqa \
        --dvqa_max 10000 --figureqa_max 5000 \
        --output_dir experiments/results/10_mixed_scr \
        --epochs 2 --lr 5e-4 --alpha 0.1 \
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
enable_scr_on_model = _sfa_mod.enable_scr_on_model
collect_scr_attn = _sfa_mod.collect_scr_attn

_density = import_module("01_density_estimator")
DensityEstimator = _density.DensityEstimator

_finetune = import_module("03_sfa_finetune")
collate_fn_factory = _finetune.collate_fn_factory
compute_vl_loss = _finetune.compute_vl_loss
save_sfa_weights = _finetune.save_sfa_weights

_scr_losses = import_module("15_scr_losses")
create_density_mask = _scr_losses.create_density_mask
compute_entropy_loss = _scr_losses.compute_entropy_loss


# ─── Dataset ────────────────────────────────────────────

class CauldronMixedDataset(Dataset):
    """Multi-dataset loader for cauldron-format JSONL files.

    All datasets use the same format:
        {"image": "images/image_XXXXXX.jpg", "question": "...", "answer": "..."}

    Large datasets (DVQA, FigureQA) are subsampled.
    """

    DEFAULT_MAX_SAMPLES = {
        "chartqa": None,
        "docvqa": None,
        "infographic_vqa": None,
        "dvqa": 10000,
        "figureqa": 5000,
    }

    def __init__(self, cauldron_root, datasets=None, max_overrides=None,
                 max_samples_total=None, input_size=448, seed=42):
        """
        Args:
            cauldron_root: Path to cauldron_data/ directory
            datasets: List of dataset names to include
            max_overrides: Dict of {dataset_name: max_samples} overrides
            max_samples_total: Global cap on total samples (for debugging)
            input_size: Image resize target
            seed: Random seed for subsampling
        """
        self.input_size = input_size
        self.samples = []
        self.dataset_counts = {}

        if datasets is None:
            datasets = list(self.DEFAULT_MAX_SAMPLES.keys())
        if max_overrides is None:
            max_overrides = {}

        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        rng = random.Random(seed)

        for ds_name in datasets:
            ds_dir = os.path.join(cauldron_root, ds_name)
            jsonl_path = os.path.join(ds_dir, "output_train.jsonl")

            if not os.path.exists(jsonl_path):
                print(f"  [WARNING] {jsonl_path} not found, skipping {ds_name}")
                continue

            # Load all samples
            ds_samples = []
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    ds_samples.append({
                        "image_path": os.path.join(ds_dir, item["image"]),
                        "question": item["question"],
                        "answer": str(item["answer"]),
                        "dataset_type": ds_name,
                    })

            # Subsample if needed
            max_s = max_overrides.get(ds_name, self.DEFAULT_MAX_SAMPLES.get(ds_name))
            if max_s is not None and len(ds_samples) > max_s:
                ds_samples = rng.sample(ds_samples, max_s)

            self.dataset_counts[ds_name] = len(ds_samples)
            self.samples.extend(ds_samples)
            print(f"  [{ds_name}] {len(ds_samples)} samples loaded")

        # Global cap (for debugging)
        if max_samples_total and len(self.samples) > max_samples_total:
            rng2 = random.Random(seed)
            self.samples = rng2.sample(self.samples, max_samples_total)

        print(f"  [Total] {len(self.samples)} samples from {len(self.dataset_counts)} datasets")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        from PIL import Image
        try:
            image = Image.open(item["image_path"]).convert("RGB")
            pixel_values = self.transform(image)
        except Exception:
            pixel_values = torch.zeros(3, self.input_size, self.input_size)

        return {
            "pixel_values": pixel_values,
            "question": item["question"],
            "answer": item["answer"],
            "dataset_type": item["dataset_type"],
        }


# ─── Helpers ────────────────────────────────────────────

def load_density_estimator(checkpoint_path, device="cuda"):
    """Load pretrained density estimator (frozen)."""
    model = DensityEstimator()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded density estimator from {checkpoint_path}")
    return model


def compute_block_ids_batch(density_model, pixel_values, num_blocks=16):
    """Run density estimator on a batch and convert to block_ids."""
    with torch.no_grad():
        density_maps = density_model(pixel_values.float())  # [B, 1, 28, 28]
    block_ids_list = []
    for i in range(density_maps.shape[0]):
        dm = density_maps[i]
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


def parse_scr_layers(layers_str):
    """Parse comma-separated layer indices."""
    return [int(x.strip()) for x in layers_str.split(",")]


# ─── Training ──────────────────────────────────────────

def train_mixed_scr(args):
    """Mixed-dataset SFA+SCR training."""
    dtype = torch.bfloat16
    device = args.device

    # GPU strategy
    num_gpus = torch.cuda.device_count()
    use_split = num_gpus >= 2

    if use_split:
        vision_device = device
        llm_device = "cuda:1" if device == "cuda:0" else "cuda:0"
    else:
        vision_device = device
        llm_device = device

    scr_layers = parse_scr_layers(args.scr_layers)

    print("=" * 60)
    print("Mixed-Dataset SFA+SCR Training")
    print("=" * 60)

    # 1. Load model
    model, tokenizer = load_internvl(
        args.model_path, device=vision_device, dtype=dtype,
        quantize_llm=(not use_split), split_gpu=use_split,
    )

    # 2. Patch with SFA (fresh initialization, no pretrained SFA checkpoint)
    model = patch_internvit_with_sfa(model)

    # 3. Load density estimator
    density_model = load_density_estimator(args.density_checkpoint, device=vision_device)

    # 4. Enable SCR attention storage
    enable_scr_on_model(model, scr_layers)
    print(f"  SCR enabled on layers: {scr_layers}")

    # 5. Freeze backbone (SFA-only training)
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

    print(f"  Frozen {frozen_count} backbone params, kept {sfa_count} SFA params trainable")

    img_ctx_id = _resolve_img_context_token_id(model, tokenizer)
    num_img_token = getattr(model, "num_image_token", 256)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.4f}%)")

    # 6. Dataset
    max_overrides = {}
    if args.dvqa_max is not None:
        max_overrides["dvqa"] = args.dvqa_max
    if args.figureqa_max is not None:
        max_overrides["figureqa"] = args.figureqa_max

    print(f"\nLoading datasets: {args.datasets}")
    train_dataset = CauldronMixedDataset(
        args.cauldron_root,
        datasets=args.datasets,
        max_overrides=max_overrides,
        max_samples_total=args.max_samples,
        input_size=448,
        seed=args.seed,
    )

    collate_fn = collate_fn_factory(tokenizer, img_ctx_id, num_img_token)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
        collate_fn=collate_fn, drop_last=True,
    )

    # 7. Optimizer + scheduler
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

    # 8. Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "train_log.json")
    logs = []
    best_loss = float("inf")
    global_step = 0
    oom_count = 0

    eff_bs = args.batch_size * args.grad_accum
    print_interval = args.grad_accum * 25  # less frequent for larger dataset

    print(f"\n  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x grad_accum {args.grad_accum} = {eff_bs}")
    print(f"  Steps/epoch: {len(train_loader)}")
    print(f"  Total optimizer steps: {total_steps}")
    print(f"  LR: {args.lr}")
    print(f"  Alpha (entropy weight): {args.alpha}")
    print(f"  SCR layers: {scr_layers}")
    print(f"  Mode: {'2-GPU split' if use_split else 'single GPU'}")
    print(f"  Dataset composition: {train_dataset.dataset_counts}")
    print()

    for epoch in range(args.epochs):
        model.train()
        if hasattr(model, "language_model"):
            model.language_model.eval()

        epoch_loss_task = 0
        epoch_loss_entropy = 0
        epoch_loss_total = 0
        epoch_steps = 0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            try:
                pixel_values = batch["pixel_values"].to(device=vision_device, dtype=dtype)

                # 1. Density -> block_ids + density_mask
                block_ids_list = compute_block_ids_batch(
                    density_model, pixel_values, num_blocks=args.num_blocks
                )
                block_ids = block_ids_list[0].to(vision_device)
                set_block_ids_on_model(model, block_ids)

                # Density mask for entropy loss
                density_mask = create_density_mask(
                    density_model, pixel_values,
                    threshold=args.density_threshold,
                ).to(vision_device)

                # 2. Task loss
                with torch.amp.autocast("cuda", dtype=dtype):
                    loss_task = compute_vl_loss(
                        model, batch, vision_device, dtype, llm_device=llm_device,
                    )

                    # 3. Collect attention from SCR layers
                    attn_list = collect_scr_attn(model, scr_layers)

                    # 4. Entropy loss on text-dense patches
                    if attn_list and density_mask.any():
                        loss_entropy = compute_entropy_loss(attn_list, density_mask)
                        if loss_entropy.item() < 1.0:
                            loss_entropy = torch.tensor(0.0, device=vision_device)
                    else:
                        loss_entropy = torch.tensor(0.0, device=vision_device)

                    # 5. Combined loss
                    loss = (loss_task + args.alpha * loss_entropy) / args.grad_accum

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

            epoch_loss_task += loss_task.item()
            epoch_loss_entropy += loss_entropy.item()
            epoch_loss_total += loss.item() * args.grad_accum
            epoch_steps += 1

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if (step + 1) % print_interval == 0:
                avg_total = epoch_loss_total / epoch_steps
                avg_task = epoch_loss_task / epoch_steps
                avg_ent = epoch_loss_entropy / epoch_steps
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                mem0 = torch.cuda.memory_allocated(vision_device) / 1024**3
                if use_split:
                    mem1 = torch.cuda.memory_allocated(llm_device) / 1024**3
                    mem_str = f"G0:{mem0:.1f}GB G1:{mem1:.1f}GB"
                else:
                    mem_str = f"mem:{mem0:.1f}GB"
                print(f"  [Epoch {epoch+1}] step {step+1}/{len(train_loader)} | "
                      f"total: {avg_total:.4f} task: {avg_task:.4f} ent: {avg_ent:.4f} | "
                      f"lr: {lr_now:.2e} | {mem_str} | {elapsed:.0f}s")

        # Epoch summary
        avg_total = epoch_loss_total / max(epoch_steps, 1)
        avg_task = epoch_loss_task / max(epoch_steps, 1)
        avg_ent = epoch_loss_entropy / max(epoch_steps, 1)
        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch+1}/{args.epochs}: total={avg_total:.4f} "
              f"task={avg_task:.4f} entropy={avg_ent:.4f} ({elapsed:.0f}s)")

        log_entry = {
            "epoch": epoch + 1,
            "loss_total": avg_total,
            "loss_task": avg_task,
            "loss_entropy": avg_ent,
            "lr": scheduler.get_last_lr()[0],
            "time_sec": elapsed,
            "oom_count": oom_count,
        }
        logs.append(log_entry)

        if avg_total < best_loss:
            best_loss = avg_total
            save_sfa_weights(model, os.path.join(args.output_dir, "best.pth"))
            print(f"  Saved best checkpoint (loss={best_loss:.4f})")

        save_sfa_weights(model, os.path.join(args.output_dir, "latest.pth"))

        with open(log_file, "w") as f:
            json.dump({
                "config": vars(args),
                "dataset_sizes": train_dataset.dataset_counts,
                "logs": logs,
            }, f, indent=2)

    # Clear state
    set_block_ids_on_model(model, None)

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Mixed-Dataset SFA+SCR Training")
    parser.add_argument("--model_path", default="/NetDisk/j_son/internvl_35")
    parser.add_argument("--cauldron_root", default="/NetDisk/juyeon/train/cauldron_data")
    parser.add_argument("--datasets", nargs="+",
                        default=["chartqa", "docvqa", "infographic_vqa", "dvqa", "figureqa"])
    parser.add_argument("--dvqa_max", type=int, default=10000)
    parser.add_argument("--figureqa_max", type=int, default=5000)
    parser.add_argument("--density_checkpoint", default="experiments/results/01_density/best.pth")
    parser.add_argument("--output_dir", default="experiments/results/10_mixed_scr")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None, help="Global sample cap (for debugging)")
    parser.add_argument("--num_blocks", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.1, help="Entropy loss weight")
    parser.add_argument("--scr_layers", default="18,19,20,21,22,23")
    parser.add_argument("--density_threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_mixed_scr(args)


if __name__ == "__main__":
    main()
