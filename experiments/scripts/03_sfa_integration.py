"""
Experiment 3: SFA를 InternVL3.5에 통합하여 Fine-tuning
======================================================
InternViT-300M의 attention layer를 SFAAttention으로 교체하고
vision encoder + projector만 학습합니다 (LLM frozen).

논문 Table 1: Ablation (SFA only, ADAT only, SFA+ADAT, Full) 결과 생성.

Usage:
    conda activate docmllm
    # Quick test (10 samples)
    python experiments/scripts/03_sfa_integration.py \
        --mode test \
        --model_path /NetDisk/j_son/Model_original/InternVL_35

    # Full training
    python experiments/scripts/03_sfa_integration.py \
        --mode train \
        --model_path /NetDisk/j_son/Model_original/InternVL_35 \
        --train_data /NetDisk/juyeon/train/chartQA/ChartQA\ Dataset/train \
        --output_dir experiments/results/03_sfa_train \
        --epochs 3

    # Evaluation
    python experiments/scripts/03_sfa_integration.py \
        --mode eval \
        --model_path experiments/results/03_sfa_train/checkpoint \
        --output_dir experiments/results/03_sfa_eval
"""

import argparse
import json
import os
import sys
import copy

import torch
import torch.nn as nn

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from importlib import import_module
_sfa = import_module("02_sfa_module")
SFAAttention = _sfa.SFAAttention
StructuralBias = _sfa.StructuralBias
compute_attention_entropy = _sfa.compute_attention_entropy


def patch_internvit_with_sfa(model, num_patches_h=32, num_patches_w=32):
    """
    InternViT의 self-attention을 SFA로 교체합니다.

    InternViT의 attention layer 구조:
        InternVisionModel.encoder.layers[i].attn

    교체 전략:
    - QKV weight를 SFA의 qkv에 복사
    - proj weight를 SFA의 proj에 복사
    - structural bias는 새로 초기화 (small init → pretrained attention 유지)
    """
    vision_model = None

    # InternVL3.5 구조에서 vision model 찾기
    if hasattr(model, "vision_model"):
        vision_model = model.vision_model
    elif hasattr(model, "model") and hasattr(model.model, "vision_model"):
        vision_model = model.model.vision_model
    else:
        print("[WARN] Cannot find vision_model in the model. Searching...")
        for name, module in model.named_modules():
            if "vision" in name.lower() and hasattr(module, "encoder"):
                vision_model = module
                print(f"  Found vision model at: {name}")
                break

    if vision_model is None:
        raise RuntimeError("Could not find InternViT vision model")

    encoder = vision_model.encoder
    replaced = 0

    for i, layer in enumerate(encoder.layers):
        attn = layer.attn
        if not hasattr(attn, "qkv"):
            print(f"  [SKIP] Layer {i}: no qkv attribute")
            continue

        dim = attn.qkv.in_features
        num_heads = attn.num_heads if hasattr(attn, "num_heads") else 16

        # Create SFA attention
        sfa = SFAAttention(
            dim=dim,
            num_heads=num_heads,
            num_patches_h=num_patches_h,
            num_patches_w=num_patches_w,
        )

        # Copy pretrained weights
        with torch.no_grad():
            sfa.qkv.weight.copy_(attn.qkv.weight)
            if attn.qkv.bias is not None:
                sfa.qkv.bias.copy_(attn.qkv.bias)
            sfa.proj.weight.copy_(attn.proj.weight)
            if attn.proj.bias is not None:
                sfa.proj.bias.copy_(attn.proj.bias)
            if hasattr(attn, 'proj_drop'):
                sfa.proj_drop.p = attn.proj_drop.p

        # Move to same device
        device = next(attn.parameters()).device
        sfa = sfa.to(device=device, dtype=next(attn.parameters()).dtype)

        # Replace
        layer.attn = sfa
        replaced += 1

    print(f"Replaced {replaced} attention layers with SFA")

    # Freeze LLM, unfreeze vision encoder + projector
    for name, param in model.named_parameters():
        if "vision" in name or "projector" in name or "mlp1" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    return model


def test_sfa_integration(model_path):
    """SFA 통합 테스트 — forward pass 확인"""
    import json as _json
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    from safetensors.torch import load_file as load_safetensors

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True)

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = _json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    state_dict = {}
    for shard_file in shard_files:
        print(f"  Loading {shard_file}...")
        state_dict.update(load_safetensors(os.path.join(model_path, shard_file)))
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()

    print("\nBefore SFA:")
    print(f"  Vision model type: {type(model.vision_model).__name__}")

    # Patch with SFA
    model = patch_internvit_with_sfa(model)

    print("\nAfter SFA:")
    # Check one layer
    layer0_attn = model.vision_model.encoder.layers[0].attn
    print(f"  Layer 0 attn type: {type(layer0_attn).__name__}")
    print(f"  Has structural_bias: {hasattr(layer0_attn, 'structural_bias')}")

    # Test inference
    from PIL import Image
    import numpy as np
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    test_img = Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))
    transform = T.Compose([
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    pixel_values = transform(test_img).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)

    print("\nTesting inference...")
    try:
        generation_config = dict(max_new_tokens=50, do_sample=False)
        response = model.chat(tokenizer, pixel_values, "Describe this image.", generation_config)
        print(f"  Response: {response[:100]}")
        print("  SFA integration test PASSED")
    except Exception as e:
        print(f"  [ERROR] Inference failed: {e}")
        import traceback; traceback.print_exc()
        print("  SFA integration test FAILED — need to adjust forward() signature")

    return model


def extract_attention_maps(model, image, tokenizer):
    """
    SFA attention maps 추출 — 논문 Figure 5용.
    각 layer에서 content attention과 structural bias를 분리하여 저장.
    """
    attention_maps = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attention_maps[name] = output[1].detach().cpu()  # attn weights
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, SFAAttention):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # Forward pass
    model.chat(tokenizer, image, "Describe the chart.", max_new_tokens=10)

    # Remove hooks
    for h in hooks:
        h.remove()

    return attention_maps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "train", "eval"], required=True)
    parser.add_argument("--model_path", default="/NetDisk/j_son/Model_original/InternVL_35")
    parser.add_argument("--train_data", default=None)
    parser.add_argument("--output_dir", default="experiments/results/03_sfa")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "test":
        test_sfa_integration(args.model_path)
    elif args.mode == "train":
        print("Full training pipeline — see 04_full_training.py for distributed training")
    elif args.mode == "eval":
        print("Evaluation — reuses 00_baseline_eval.py with patched model")


if __name__ == "__main__":
    main()
