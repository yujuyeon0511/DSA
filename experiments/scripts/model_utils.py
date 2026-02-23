"""
공용 InternVL3.5 모델 로딩 유틸리티.
Meta tensor 이슈를 회피하여 안정적으로 로드합니다.

로딩 모드:
  - full: 전체 모델을 단일 GPU에 bf16 로드 (추론 전용, ~17GB)
  - quantized: LLM을 4-bit 양자화, vision encoder는 bf16 유지 (학습용, ~8GB)
"""

import json
import os

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoConfig, AutoModel, AutoTokenizer


def load_internvl(model_path, device="cuda", dtype=torch.bfloat16, quantize_llm=False,
                   split_gpu=False):
    """
    InternVL3.5-8B 모델을 meta tensor 이슈 없이 로드합니다.

    Args:
        model_path: 모델 경로
        device: 로드할 디바이스
        dtype: 데이터 타입 (bf16)
        quantize_llm: True이면 LLM을 4-bit로 양자화 (단일 GPU 학습용)
        split_gpu: True이면 2-GPU 모델 병렬 (vision→cuda:0, LLM→cuda:1)

    Returns:
        model, tokenizer
    """
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if quantize_llm and not split_gpu:
        return _load_internvl_quantized(model_path, config, tokenizer, device, dtype)

    model = AutoModel.from_config(config, trust_remote_code=True)

    # Load state dict from safetensors shards
    state_dict = _load_safetensors_shards(model_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    del state_dict

    if split_gpu:
        return _move_to_split_gpu(model, tokenizer, dtype)

    model = model.to(device=device, dtype=dtype).eval()
    print("Model loaded.")
    return model, tokenizer


def _load_safetensors_shards(model_path):
    """safetensors shard 파일들을 로드하여 state_dict 반환."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    state_dict = {}
    for shard_file in shard_files:
        print(f"  Loading {shard_file}...")
        state_dict.update(load_safetensors(os.path.join(model_path, shard_file)))
    return state_dict


def _move_to_split_gpu(model, tokenizer, dtype):
    """
    2-GPU 모델 병렬: Vision encoder + projector → cuda:0, LLM → cuda:1.
    양자화 불필요 (80GB 총 VRAM). batch_size 증가 가능.
    """
    print("  [Split GPU mode] Vision → cuda:0, LLM → cuda:1")

    model = model.to(dtype=dtype)

    # Vision encoder + projector → GPU 0 (trainable)
    model.vision_model = model.vision_model.to("cuda:0")
    if hasattr(model, "mlp1"):
        model.mlp1 = model.mlp1.to("cuda:0")

    # LLM → GPU 1 (frozen)
    model.language_model = model.language_model.to("cuda:1")

    # Remaining top-level params/buffers
    for name, param in model.named_parameters():
        if param.device.type == "cpu":
            if "vision" in name or "mlp1" in name:
                param.data = param.data.to("cuda:0")
            else:
                param.data = param.data.to("cuda:1")
    for name, buf in model.named_buffers():
        if buf.device.type == "cpu":
            if "vision" in name or "mlp1" in name:
                buf.data = buf.data.to("cuda:0")
            else:
                buf.data = buf.data.to("cuda:1")

    model.eval()

    mem0 = torch.cuda.memory_allocated("cuda:0") / 1024**3
    mem1 = torch.cuda.memory_allocated("cuda:1") / 1024**3
    print(f"  GPU 0 (vision): {mem0:.1f} GB")
    print(f"  GPU 1 (LLM):    {mem1:.1f} GB")
    print("Model loaded (split GPU).")
    return model, tokenizer


def _load_internvl_quantized(model_path, config, tokenizer, device, dtype):
    """
    LLM을 4-bit NF4 양자화하여 로드. Vision encoder + projector는 bf16 유지.
    학습 시 ~8GB VRAM으로 A100-40GB에서 fine-tuning 가능.
    """
    from transformers import BitsAndBytesConfig

    print("  [Quantized mode] LLM → 4-bit NF4, Vision → bf16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load state dict first
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    state_dict = {}
    for shard_file in shard_files:
        print(f"  Loading {shard_file}...")
        state_dict.update(load_safetensors(os.path.join(model_path, shard_file)))

    # Separate vision/projector weights vs LLM weights
    vision_state = {}
    llm_state = {}
    for k, v in state_dict.items():
        if k.startswith("vision_model.") or k.startswith("mlp1."):
            vision_state[k] = v
        else:
            llm_state[k] = v
    del state_dict

    # Build model on CPU first
    model = AutoModel.from_config(config, trust_remote_code=True)

    # Load all weights on CPU
    full_state = {**vision_state, **llm_state}
    missing, unexpected = model.load_state_dict(full_state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    del full_state

    # Quantize LLM layers in-place
    import bitsandbytes as bnb

    quantized_count = 0
    for name, module in model.language_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.language_model
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)

            in_features = module.in_features
            out_features = module.out_features
            has_bias = module.bias is not None

            # Create 4-bit linear
            q_linear = bnb.nn.Linear4bit(
                in_features, out_features,
                bias=has_bias,
                quant_type="nf4",
                compute_dtype=dtype,
            )
            # Copy weight data then quantize
            q_linear.weight = bnb.nn.Params4bit(
                module.weight.data,
                requires_grad=False,
                quant_type="nf4",
                compress_statistics=True,
            )
            if has_bias:
                q_linear.bias = torch.nn.Parameter(module.bias.data, requires_grad=False)

            setattr(parent, child_name, q_linear)
            quantized_count += 1

    print(f"  Quantized {quantized_count} LLM linear layers to 4-bit")

    # Move vision encoder + projector to GPU in bf16
    model.vision_model = model.vision_model.to(device=device, dtype=dtype)
    if hasattr(model, "mlp1"):
        model.mlp1 = model.mlp1.to(device=device, dtype=dtype)

    # Move LLM to GPU (quantized, much smaller)
    model.language_model = model.language_model.to(device)

    # Move any remaining top-level params
    for name, param in model.named_parameters():
        if param.device.type == "cpu":
            param.data = param.data.to(device)
    for name, buf in model.named_buffers():
        if buf.device.type == "cpu":
            buf.data = buf.data.to(device)

    model.eval()

    # Memory report
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    print(f"  GPU memory after load: {allocated:.1f} GB")
    print("Model loaded (quantized).")
    return model, tokenizer


def load_image_single_tile(image_path, input_size=448):
    """단일 타일 이미지 전처리 (448×448)"""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from PIL import Image

    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    image = Image.open(image_path).convert("RGB")
    pixel_values = transform(image).unsqueeze(0)
    return pixel_values


def run_chat(model, tokenizer, pixel_values, question, max_new_tokens=256, device="cuda", dtype=torch.bfloat16):
    """InternVL chat 호출"""
    pixel_values = pixel_values.to(device=device, dtype=dtype)
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response.strip()
