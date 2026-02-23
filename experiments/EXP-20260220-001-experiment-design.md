# Experiment Design: Structure-Factorized Attention (SFA) for Document-Centric MLLMs

**Experiment ID**: EXP-20260220-001
**Date**: 2026-02-20
**Author**: Juyeon
**Status**: In Progress (Baseline 완료, SFA Fine-tuning 스크립트 완료, 실행 대기)
**Last Updated**: 2026-02-20 (v2 — cross-architecture 요구사항 반영)

> 이 문서가 **유일한 실험 마스터 문서**입니다.
> 실험 설계, 환경 설정, 시각화 규격, 실행 로그, TODO 모두 이 파일에 기록합니다.

---

## 1. 연구 목표

문서/차트 특화 멀티모달 태스크에서 기존 ViT 기반 vision encoder의 구조적 한계를 해결하기 위해
**Structure-Factorized Attention (SFA)** + **Adaptive Density-Aware Tokenization (ADAT)** 을 제안하고
InternVL3.5-8B 위에서 검증한다.

**핵심 가설:**
1. 문서 이미지에서 attention에 structural bias (row/col/block)를 주입하면 grounding이 안정화된다
2. 텍스트 밀집 영역에 토큰을 집중 배분하면 동일 budget 대비 정확도가 향상된다
3. 위 두 방법으로 attention entropy가 감소하고 hallucination이 줄어든다

**방법론 유의미성 조건 (Cross-Architecture Generalization):**
> SFA/ADAT가 **특정 vision encoder + 특정 LLM** 조합에서만 효과가 있다면 novelty가 약함.
> **어떤 vision encoder (InternViT, CLIP, SigLIP)** 에 적용하더라도,
> **어떤 LLM decoder (Qwen, LLaMA, InternLM)** 와 결합하더라도
> document-centric 벤치마크에서 실질적 성능 향상이 있어야 함.
>
> → Phase 5에 cross-architecture 실험 추가 (§10 참조)

---

## 2. 환경 설정

### 2.1 모델 세팅

| 항목 | 값 |
|------|-----|
| Base Model | InternVL3.5-8B |
| 모델 경로 (실제 사용) | **`/NetDisk/j_son/internvl_35/`** (48GB, conversation.py 포함) |
| 모델 경로 (원본) | `/NetDisk/j_son/Model_original/InternVL_35` |
| Vision Encoder | InternViT-300M-448px (24 layers, dim=1024, 16 heads) |
| Patch Config | 448px / 14px patch = **32×32 grid + 1 CLS = 1025 tokens** |
| LLM | InternLM2.5-7B-Chat (frozen) |
| 수정 범위 | Vision encoder attention + projector만 학습 |
| 모델 로딩 | `AutoModel.from_config()` + safetensors 수동 로딩 (**`from_pretrained` 사용 불가 — meta tensor 이슈**) |

### 2.2 GPU / Software

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA A100-PCIE-40GB × 2 |
| CUDA | 12.1 |
| 추론 dtype | bfloat16 |
| Python | 3.10.19 |
| PyTorch | 2.4.0+cu121 |
| transformers | 5.1.0 |
| flash_attn | 2.8.3 |
| matplotlib | 3.10.8 |
| opencv | 4.11.0.86 (headless) |
| Conda env | `docmllm` |

```bash
conda activate docmllm
# 추가 설치 (1회)
pip install seaborn scipy scikit-learn
```

### 2.3 비교 모델 (Ablation)

| # | 모델 | 설명 |
|---|------|------|
| A | **Baseline** | Original InternVL3.5-8B (수정 없음) |
| B | **+ SFA only** | Attention에 structural bias φ(s_i, s_j) 추가 |
| C | **+ ADAT only** | Adaptive density-aware tokenization 적용 |
| D | **+ SFA + ADAT** | 두 모듈 동시 적용 |
| E | **+ Full (SFA + ADAT + SCR)** | Entropy/Stability regularization 추가 |

### 2.3.1 Cross-Architecture 검증 모델

SFA의 범용성을 입증하기 위해 아래 조합에서도 실험:

| Vision Encoder | LLM Decoder | 기반 모델 | 비고 |
|----------------|-------------|-----------|------|
| InternViT-300M | InternLM2.5-7B | InternVL3.5-8B | **Primary (위 A~E)** |
| SigLIP-SO400M | Qwen2.5-7B | Qwen2.5-VL-7B | SigLIP + Qwen 조합 |
| CLIP-ViT-L/14 | Vicuna-7B / LLaMA3-8B | LLaVA-OV-7B | CLIP + LLaMA 계열 |

**핵심 실험**: 각 조합에서 `Baseline` vs `+ SFA` 의 ChartQA Relaxed Accuracy 비교
- 3개 모델 모두에서 SFA 적용 시 성능 향상 → **architecture-agnostic contribution** 주장 가능
- 1개 이상에서 향상 없음 → 해당 encoder 특성 분석 필요

### 2.4 기술적 이슈 해결 기록

| 이슈 | 원인 | 해결 |
|------|------|------|
| `conversation.py` 누락 | `Model_original` 디렉토리에 파일 없음 | 모델 경로를 `/NetDisk/j_son/internvl_35/`로 변경 |
| `torch_dtype` deprecated | transformers 5.x에서 제거됨 | `dtype` 파라미터 사용 |
| Meta tensor `RuntimeError` | `device_map="auto"` 시 InternViT의 `torch.linspace().item()` 호출 | `from_config()` + safetensors shard 수동 로딩 (`model_utils.py`) |
| `model.chat()` API | `max_new_tokens`가 keyword arg가 아님 | `generation_config = dict(...)` 형태로 positional arg 전달 |
| Flash attention → attention weight 미노출 | InternViT 기본값이 flash attn | `use_flash_attn=False` + QKV hook으로 수동 계산 |
| Patch grid 28 vs 32 | 448/14 = 32 (not 28) | 모든 모듈의 `num_patches_h/w`를 32로 수정 |
| CLS token 처리 | N=1025 = 1 CLS + 32×32 spatial | structural bias를 `[:, :, 1:, 1:]`에만 적용, entropy 계산 시 `[:, 1:]` |
| SFA forward 반환값 불일치 | InternAttention → 단일 텐서, SFA → (out, attn) 튜플 | 단일 텐서 반환, attn은 `_last_attn_weights`에 저장 |
| `proj_drop` 누락 | InternAttention에 있으나 SFA에 없음 | `nn.Dropout(0.0)` 추가 |
| VL loss gradient 끊김 | `model.forward()` 내부에서 `img_context_token_id` 미설정 + token 수 불일치 | 수동 `compute_vl_loss()` 구현: `.clone()` + `*0.0 + vit_embeds` gradient trick |
| `img_context_token_id` 미설정 | `__init__`에서 미설정, `chat()`에서만 동적 설정 | `_resolve_img_context_token_id()` 헬퍼로 tokenizer에서 검색 후 model attribute 설정 |

---

## 3. 실험 단계별 설계

### Step 0: Baseline Evaluation ✅ 완료

**목적**: 수정 전 InternVL3.5-8B의 벤치마크 기준 수치 확보
**스크립트**: `experiments/scripts/00_baseline_eval.py`

```bash
python experiments/scripts/00_baseline_eval.py \
    --model_path /NetDisk/j_son/internvl_35 \
    --output_dir experiments/results/00_baseline \
    --benchmarks chartqa --max_samples 200
```

**결과**:
| 벤치마크 | Metric | Score | Samples | Time |
|----------|--------|-------|---------|------|
| ChartQA | Relaxed Accuracy | **0.6200** | 200 | 150.5s |

- Single-tile (448×448) 추론, dynamic_preprocess 미적용
- 오답 예시: 소수점 수치(0.57→10.04), 미세 차이(0.08→0.02) — **숫자 grounding 취약점 확인**

---

### Step 1: Text Density Estimator Training ✅ 완료

**목적**: 문서 이미지의 텍스트 밀집도를 예측하는 lightweight CNN 학습
**스크립트**: `experiments/scripts/01_density_estimator.py`

**Pseudo Label**: Canny edge + adaptive threshold → Gaussian blur → 28×28 density map
**모델**: 6-layer CNN (3→32→64→128→64→32→1), 186K params

```bash
python experiments/scripts/01_density_estimator.py \
    --mode train \
    --data_dirs "/NetDisk/juyeon/train/chartQA/ChartQA Dataset/train/png" /NetDisk/juyeon/train/dvqa/images \
    --output_dir experiments/results/01_density --epochs 10 --batch_size 64 --max_images 20000
```

**결과**:
| 항목 | 값 |
|------|-----|
| Train/Val | 19,000 / 1,000 |
| Best Val Loss | **0.001728** (Epoch 10) |
| 산출물 | `best.pth`, `final.pth`, `visualizations/density_*.png` (20장) |

---

### Step 2: SFA Module Test ✅ 완료

**목적**: Structure-Factorized Attention 모듈 단독 동작 검증
**스크립트**: `experiments/scripts/02_sfa_module.py`

**SFA 수식**:
```
S_ij = (Q_i K_j^T) / sqrt(d) + φ(s_i, s_j)
φ(s_i, s_j) = w_row·[row_i==row_j] + w_col·[col_i==col_j] + w_dist·(-manhattan(i,j)) + block_embed(b_i)^T·block_embed(b_j)
```

**결과**:
| 항목 | 값 |
|------|-----|
| Forward | [2, 784, 1024] → [2, 784, 1024] OK |
| Attention Entropy | 6.6089 |
| Structural Bias Params | **304** per layer (0.007%) |

---

### Step 3: SFA → InternVL3.5 Integration ✅ 완료

**목적**: InternViT의 self-attention을 SFA로 교체 후 inference 동작 확인
**스크립트**: `experiments/scripts/03_sfa_integration.py`

**교체 전략**:
1. `InternVisionModel.encoder.layers[i].attn` → `SFAAttention`
2. QKV/proj weight 복사 (pretrained 유지), structural bias small init (std=0.02)
3. LLM frozen, vision encoder + projector만 trainable

**결과**:
| 항목 | 값 |
|------|-----|
| SFA 교체 layers | **24 / 24** (전체) |
| Trainable | 337,590,400 / 8,528,325,760 (**4.0%**) |
| Inference | **PASSED** |

---

### Step 4: Attention Entropy Analysis (Baseline) ✅ 완료

**목적**: text-dense vs sparse region의 attention entropy 차이 측정
**스크립트**: `experiments/scripts/04_attention_analysis.py --mode entropy`

**방법**: flash attn 비활성화 → QKV hook으로 마지막 4 layers 캡처 → density map 기반 region 분리

**결과**:
| 측정 영역 | Entropy |
|-----------|---------|
| Text-dense | **4.3322** |
| Sparse | **4.4377** |
| Ratio | **0.98x** |

→ **구조적 바이어스 부재 확인** — SFA 적용 후 text region entropy 감소 시 가설 검증 성공

---

### Step 5: Hallucination Rate Analysis (Baseline) ✅ 완료

**스크립트**: `experiments/scripts/04_attention_analysis.py --mode hallucination`

**결과**:
| 지표 | 값 |
|------|-----|
| Accuracy | **0.6500** |
| Hallucination Rate | **0.2550** (51/200) |
| Wrong Answer Rate | 0.0950 (19/200) |

→ 오류의 **73%가 숫자 hallucination** — 구조적 grounding 부재가 주 원인

---

### Step 6: Token Efficiency Curve ✅ Placeholder 완료

**스크립트**: `experiments/scripts/04_attention_analysis.py --mode token_efficiency`
**산출물**: `token_efficiency_curve.{pdf,png}` — ADAT 구현 후 실 데이터로 교체 필요

---

### Step 7: Attention Heatmap (Figure 3) — 구현 필요

**목적**: Baseline vs SFA attention 분포를 동일 이미지/질의에 대해 시각적 비교
**스크립트**: `experiments/scripts/06_attention_heatmap.py` (신규)

**입력**:
- 샘플: `experiments/figures/sample_images/chartqa_sample.png` (원본: `00339007006077.png`)
- 질의: `"What is the value for Haiti?"`
- Layers: 20, 21, 22, 23

**방법**:
1. `use_flash_attn = False` → QKV hook으로 attention weight 캡처
2. Head 평균 → CLS→spatial attention (0번→1:N) → 32×32 reshape
3. bilinear interpolation → 원본 해상도 upscale
4. `YlOrRd` colormap, alpha=0.5 오버레이

```bash
# Baseline
python experiments/scripts/06_attention_heatmap.py \
    --model_type baseline --model_path /NetDisk/j_son/internvl_35 \
    --image experiments/figures/sample_images/chartqa_sample.png \
    --question "What is the value for Haiti?" --layers 20 21 22 23 \
    --output_dir experiments/figures/fig3_attention

# SFA (fine-tuning 후)
python experiments/scripts/06_attention_heatmap.py \
    --model_type sfa --model_path /NetDisk/j_son/internvl_35 \
    --sfa_checkpoint experiments/results/03_sfa/sfa_weights.pth \
    --image experiments/figures/sample_images/chartqa_sample.png \
    --question "What is the value for Haiti?" --layers 20 21 22 23 \
    --output_dir experiments/figures/fig3_attention
```

---

### Step 8: Motivation Figure (Figure 1) — 구현 필요

**스크립트**: `experiments/scripts/05_figure_motivation.py` (신규)
**구성** (1×3): (a) Original + 14×14 grid, (b) Density overlay, (c) Adaptive patching

```bash
python experiments/scripts/05_figure_motivation.py \
    --image experiments/figures/sample_images/chartqa_sample.png \
    --density_ckpt experiments/results/01_density/best.pth \
    --output_dir experiments/figures/fig1_motivation
```

---

### Step 9: Entropy Figure (Figure 5) — 구현 필요

**스크립트**: `experiments/scripts/07_figure_entropy.py` (신규)
**구성** (1×2): (a) Violin/box plot (text vs sparse), (b) Layer-wise entropy line plot (Baseline vs SFA)

```bash
python experiments/scripts/07_figure_entropy.py \
    --baseline_data experiments/results/04_analysis/entropy_analysis.json \
    --sfa_data experiments/results/05_sfa_analysis/entropy_analysis.json \
    --output_dir experiments/figures/fig5_entropy
```

---

### Step 10: Structural Bias 시각화 (Figure 7) — SFA fine-tuning 후

**스크립트**: `experiments/scripts/09_structural_bias_viz.py` (신규)
**구성** (1×4): (a) Row bias, (b) Col bias, (c) Distance decay, (d) Combined φ

```bash
python experiments/scripts/09_structural_bias_viz.py \
    --sfa_checkpoint experiments/results/03_sfa/sfa_weights.pth \
    --grid_size 32 --output_dir experiments/figures/fig7_structural_bias
```

---

## 4. 데이터 경로

### 평가 데이터
| 벤치마크 | 경로 | Metric |
|----------|------|--------|
| ChartQA test | `/NetDisk/juyeon/train/chartQA/ChartQA Dataset/test/` | Relaxed Accuracy |
| DocVQA | `/NetDisk/juyeon/train/cauldron_data/docvqa/` | ANLS |
| PlotQA test | `/NetDisk/juyeon/train/plotqa/test/` | Relaxed Accuracy |
| DVQA val | `/NetDisk/juyeon/train/dvqa/val_easy_qa.json` | Exact Match |
| FigureQA test | `/NetDisk/juyeon/train/figureqa/no_annot_test1/` | Accuracy |
| OCRBench | HuggingFace `echo840/ocrbench` | Exact Match |

### 학습 데이터
| 용도 | 데이터 | 경로 | 규모 |
|------|--------|------|------|
| Density Estimator | ChartQA train | `/NetDisk/juyeon/train/chartQA/ChartQA Dataset/train/png/` | 18K |
| Density Estimator | DVQA | `/NetDisk/juyeon/train/dvqa/images/` | ~300K |
| Vision Fine-tuning | mllm_ready V3 | `/NetDisk/ingyu/VLM_DATA/mllm_ready/labels/V3/` | 5.5M rows |
| Vision Fine-tuning | Cauldron | `/NetDisk/juyeon/train/cauldron_data/` | 41GB |
| Vision Fine-tuning | LLaVA | `/NetDisk/juyeon/train/llava_data/` | 47GB |

---

## 5. 논문 Figure 상세 계획

### 5.1 시각화 규격

**출력 형식**: PDF (학회 제출) + PNG 300dpi (프리뷰)
**컬럼 폭**: Single 3.25in, Double 6.875in (ECCV)
**폰트 하한**: 8pt, **선 두께 하한**: 1pt

### 5.2 색상표

| 역할 | 이름 | HEX | 용도 |
|------|------|-----|------|
| **Ours (SFA)** | Research Blue | `#1A73E8` | SFA 커브, 핵심 블록 |
| **구조 바이어스** | Structural Teal | `#00897B` | Row/Col bias, φ 함수 |
| **밀도 강조** | Density Amber | `#F9AB00` | 정보 밀집 지역 |
| **Baseline** | Neutral Slate | `#70757A` | Baseline 결과 |
| **배경** | Paper White | `#F8F9FA` | 그래프 배경 |
| **텍스트** | Charcoal | `#202124` | 축 레이블 |
| + ADAT only | Coral | `#E8453C` | Ablation C |
| + SFA + ADAT | Purple | `#7B1FA2` | Ablation D |
| + Full (SCR) | Deep Blue | `#0D47A1` | Ablation E |
| Qwen2.5-VL | Pine Green | `#2E7D32` | 외부 비교 |
| LLaVA-OV | Brown | `#795548` | 외부 비교 |

### 5.3 히트맵 Colormap

| 용도 | Colormap |
|------|----------|
| Attention 히트맵 | `YlOrRd` |
| Density map | `inferno` |
| Entropy 분포 | `coolwarm` |
| Diff (SFA-Baseline) | `RdBu_r` |

### 5.4 matplotlib rcParams (모든 스크립트 공통)

```python
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "STIXGeneral"],
    "font.sans-serif": ["DejaVu Sans"], "font.size": 10, "mathtext.fontset": "stix",
    "axes.titlesize": 14, "axes.labelsize": 12, "axes.linewidth": 0.8,
    "axes.edgecolor": "#202124", "axes.labelcolor": "#202124", "axes.facecolor": "#F8F9FA",
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.fontsize": 10, "legend.framealpha": 0.9, "legend.edgecolor": "#CCCCCC",
    "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "figure.facecolor": "white",
})

COLORS = {
    "ours": "#1A73E8", "struct": "#00897B", "density": "#F9AB00",
    "baseline": "#70757A", "bg": "#F8F9FA", "text": "#202124",
    "adat": "#E8453C", "sfa_adat": "#7B1FA2", "full": "#0D47A1",
    "qwen": "#2E7D32", "llava": "#795548",
}
```

### 5.5 샘플 이미지

| 파일 | 경로 | 특징 |
|------|------|------|
| `chartqa_sample.png` | `experiments/figures/sample_images/` | 수평 바차트, 5개국 비교, 축/레이블/수치 포함 |

원본: `/NetDisk/juyeon/train/chartQA/ChartQA Dataset/test/png/00339007006077.png`

### 5.6 Figure ↔ 실험 매핑

| Figure | 내용 | 스크립트 | 상태 |
|--------|------|---------|------|
| **Fig 1**: Motivation | Uniform vs Adaptive patching 3-panel | `05_figure_motivation.py` | ✅ **스크립트 완료** |
| **Fig 2**: Architecture | 파이프라인 흐름도 | draw.io / TikZ | 수동 |
| **Fig 3**: Attention Heatmap | Baseline vs SFA 오버레이 비교 | `06_attention_heatmap.py` | ✅ **스크립트 완료** (Baseline 실행 대기) |
| **Fig 4**: Density Map | Original / Pseudo GT / Predicted | `01_density_estimator.py` | **데이터 완료** (20장) |
| **Fig 5**: Entropy | Text vs Sparse 분포 + layer-wise | `07_figure_entropy.py` | ✅ **스크립트 완료** (Baseline 실행 대기) |
| **Fig 6**: Token Efficiency | Budget sweep curve | `08_token_efficiency.py` | Placeholder 완료 |
| **Fig 7**: Structural Bias | φ 행렬 4-panel (row/col/dist/combined) | `09_structural_bias_viz.py` | SFA fine-tuning 후 |
| **Fig 8**: Cross-Arch | Encoder별 SFA 효과 Bar chart | `12_cross_arch.py` | 📋 Phase 5 |

---

## 6. 논문 Table 상세 계획

### Table 1 — Main Results (InternVL3.5 Ablation)

| 모델 | ChartQA | DocVQA | TextVQA | OCRBench | AI2D |
|------|---------|--------|---------|----------|------|
| InternVL2.5-8B | (공개 수치) | | | | |
| Qwen2.5-VL-7B | (공개 수치) | | | | |
| LLaVA-OV-7B | (공개 수치) | | | | |
| **InternVL3.5-8B (Baseline)** | **0.620** | TBD | TBD | TBD | TBD |
| + SFA | TBD | | | | |
| + ADAT | TBD | | | | |
| + SFA + ADAT | TBD | | | | |
| **+ Full (Ours)** | TBD | | | | |

### Table 1b — Cross-Architecture Generalization

| Vision Encoder → LLM | Baseline | + SFA | Δ |
|----------------------|----------|-------|---|
| InternViT → InternLM2.5 (InternVL3.5) | 0.620 | TBD | TBD |
| SigLIP → Qwen2.5 (Qwen2.5-VL) | TBD | TBD | TBD |
| CLIP-ViT-L → LLaMA3 (LLaVA-OV) | TBD | TBD | TBD |

> 3개 이상의 encoder-decoder 조합에서 일관된 향상 → "SFA is architecture-agnostic"

### Table 2 — Ablation Study

| 모듈 | ChartQA | DocVQA | Halluc Rate | Params (추가) |
|------|---------|--------|-------------|--------------|
| Baseline | 0.620 | TBD | 25.5% | +0 |
| + SFA | TBD | TBD | TBD | +7.3K (304/layer × 24) |
| + ADAT | TBD | TBD | TBD | +186K |
| + SFA + ADAT | TBD | TBD | TBD | +7.3K + 186K |
| + Full (SFA+ADAT+SCR) | TBD | TBD | TBD | 위와 동일 (SCR은 loss만) |

### Table 3 — Structural Component Study

| Row | Col | Dist | Block | ChartQA | Halluc Rate |
|-----|-----|------|-------|---------|-------------|
| | | | | 0.620 | 25.5% |
| v | | | | TBD | TBD |
| | v | | | TBD | TBD |
| v | v | | | TBD | TBD |
| v | v | v | | TBD | TBD |
| v | v | v | v | TBD | TBD |

### Table 4 — Computational Cost

| 모델 | FLOPs (G) | Latency (ms) | Vision Tokens | Total Params |
|------|-----------|-------------|---------------|-------------|
| Baseline | TBD | TBD | 1024 | 8.5B |
| + SFA | TBD | TBD | 1024 | 8.5B + 7.3K |
| + ADAT (N=512) | TBD | TBD | 512 | 8.5B + 186K |

측정: `fvcore.nn.FlopCountAnalysis`, `torch.cuda.Event` 기반

---

## 7. Appendix 시각화

### 7.1 Hallucination Case Study
- 2×4 grid: Baseline 틀리고 SFA 맞추는 4 cases
- 스크립트: `10_hallucination_cases.py`
- Baseline 데이터: `results/04_analysis/hallucination_analysis.json` (51건 숫자 hallucination)

### 7.2 Multi-Resolution Scaling
- Line plot: 해상도 {224, 448, 896, 1344} × Baseline vs SFA+ADAT

### 7.3 Density Map Gallery
- 4×5 grid, 다양한 문서 유형
- **이미 생성됨**: `results/01_density/visualizations/density_000~019.png`

---

## 8. 스크립트 목록

| # | 스크립트 | 용도 | 상태 |
|---|---------|------|------|
| -- | `model_utils.py` | 공용 모델 로딩 (meta tensor 회피) | ✅ **완료** |
| -- | `run_all.sh` | 전체 파이프라인 | ✅ **완료** |
| 00 | `00_baseline_eval.py` | Baseline 벤치마크 평가 | ✅ **실행 완료** |
| 01 | `01_density_estimator.py` | Density Estimator 학습/시각화 | ✅ **실행 완료** |
| 02 | `02_sfa_module.py` | SFA 모듈 단위 테스트 | ✅ **실행 완료** |
| 03a | `03_sfa_integration.py` | SFA→InternVL 통합 | ✅ **실행 완료** |
| 03b | `03_sfa_finetune.py` | **SFA Fine-tuning** (train/eval) | ✅ **완료** (test passed, 실행 대기) |
| 04 | `04_attention_analysis.py` | Entropy/Hallucination/Token eff. | ✅ **실행 완료** |
| 05 | `05_figure_motivation.py` | Figure 1: Motivation | ✅ **스크립트 완료** (실행 대기) |
| 06 | `06_attention_heatmap.py` | Figure 3: Attention 히트맵 | ✅ **스크립트 완료** (실행 대기) |
| 07 | `07_figure_entropy.py` | Figure 5: Entropy 그래프 | ✅ **스크립트 완료** (실행 대기) |
| 08 | `08_token_efficiency.py` | Figure 6: Token efficiency | **구현 필요** |
| 09 | `09_structural_bias_viz.py` | Figure 7: φ 시각화 | **구현 필요** |
| 10 | `10_hallucination_cases.py` | Appendix: Case study | **구현 필요** |
| 11 | `11_adat_module.py` | ADAT 모듈 | **구현 필요** |
| 12 | `12_cross_arch.py` | Cross-Architecture 실험 | 📋 Phase 5 |

---

## 9. 실행 로그

### 2026-02-20 — Baseline 전체 분석 완료

| Step | 실험 | 핵심 결과 |
|------|------|----------|
| 0 | Baseline Eval | ChartQA Acc = **0.620** (200 samples) |
| 1 | Density Estimator | Val Loss = **0.00173**, 186K params |
| 2 | SFA Module Test | Forward OK, bias params = 304 |
| 3 | SFA Integration | 24 layers, trainable **4.0%** |
| 4 | Entropy Analysis | Text/Sparse = **0.98x** (구조적 바이어스 부재) |
| 5 | Hallucination | Rate = **25.5%**, 오류의 73%가 숫자 hallucination |
| 6 | Token Efficiency | Placeholder 생성 |

**핵심 발견**:
1. Text-dense/sparse 간 entropy 차이 거의 없음 (0.98x) → **SFA 필요성 입증**
2. Baseline 오류의 73%가 숫자 hallucination → **구조적 grounding 강화 필요**

**생성된 파일**:
```
experiments/
├── results/
│   ├── 00_baseline/summary.json
│   ├── 01_density/best.pth, final.pth, visualizations/ (20장)
│   └── 04_analysis/entropy_analysis.json, hallucination_analysis.json, token_efficiency_curve.{pdf,png}
├── scripts/
│   ├── 03_sfa_finetune.py       (P2-1, 구현 중)
│   ├── 05_figure_motivation.py  (P1-1, 스크립트 완료)
│   ├── 06_attention_heatmap.py  (P1-2, 스크립트 완료)
│   └── 07_figure_entropy.py     (P1-3, 스크립트 완료)
└── figures/
    └── sample_images/chartqa_sample.png
```

### 2026-02-20 (오후) — Phase 1 스크립트 + P2-1 진행

| 작업 | 상태 | 비고 |
|------|------|------|
| P1-1 `05_figure_motivation.py` | ✅ 스크립트 완료 | 3-panel (uniform/density/adaptive) |
| P1-2 `06_attention_heatmap.py` | ✅ 스크립트 완료 | extract + compose 모드 |
| P1-3 `07_figure_entropy.py` | ✅ 스크립트 완료 | violin + layer-wise 2-panel |
| P2-1 `03_sfa_finetune.py` | ✅ 완료 | gradient fix 완료, test passed (loss 4.78→3.08) |

**P2-1 기술 이슈 → ✅ 해결**:
- ❌ 초기: `model.forward()` 직접 호출 시 `img_context_token_id` 미설정 + token 수 불일치
- ✅ 수정: `compute_vl_loss()`를 수동 구현 (`.clone()` + `*0.0 + vit_embeds` gradient trick)
- ✅ `_resolve_img_context_token_id()` 헬퍼로 model attribute 일관성 보장
- ✅ **Test PASSED**: 10 steps, loss 4.78 → 3.08 감소, structural bias weights 업데이트 확인
  - Layer 0 w_row = [0.018, -0.009, -0.002, 0.015] (non-zero → 학습 동작 정상)

---

## 10. 앞으로 해야 할 실험 (TODO)

### Phase 1: 시각화 스크립트 ✅ 스크립트 구현 완료

| ID | 작업 | 의존성 | 산출물 | 상태 |
|----|------|--------|--------|------|
| P1-1 | `05_figure_motivation.py` 실행 | density ckpt ✅ | Figure 1 | ✅ 스크립트 완료, 실행 대기 |
| P1-2 | `06_attention_heatmap.py` Baseline 실행 | 모델 ✅ | Figure 3 (Baseline half) | ✅ 스크립트 완료, 실행 대기 |
| P1-3 | `07_figure_entropy.py` Baseline 실행 | entropy data ✅ | Figure 5 (Baseline half) | ✅ 스크립트 완료, 실행 대기 |

### Phase 2: SFA Fine-tuning (핵심) — 🔧 진행 중

| ID | 작업 | 의존성 | 시간 | 산출물 | 상태 |
|----|------|--------|------|--------|------|
| P2-1 | **SFA fine-tuning 스크립트** | Step 3 ✅ | 2시간 | `03_sfa_finetune.py` | ✅ **완료** (test passed) |
| P2-2 | **SFA fine-tuning 실행** (ChartQA train) | P2-1 | **~24h** | `results/03_sfa_train/` | ⬜ |
| P2-3 | SFA 모델 ChartQA eval | P2-2 | 1시간 | Table 1 "+SFA" | ⬜ |
| P2-4 | SFA entropy 재측정 | P2-2 | 2시간 | Figure 5 완성 | ⬜ |
| P2-5 | SFA hallucination 재측정 | P2-2 | 2시간 | Table 2 "+SFA" | ⬜ |
| P2-6 | SFA attention heatmap | P2-2 | 1시간 | Figure 3 완성 | ⬜ |
| P2-7 | Structural bias 시각화 | P2-2 | 30분 | Figure 7 | ⬜ |
| P2-8 | Structural component ablation | P2-2 | 12시간 | Table 3 | ⬜ |

### Phase 3: ADAT 구현 + 통합

| ID | 작업 | 의존성 | 시간 | 산출물 | 상태 |
|----|------|--------|------|--------|------|
| P3-1 | **ADAT 모듈 구현** | Density Est. ✅ | 4시간 | `11_adat_module.py` | ⬜ |
| P3-2 | ADAT 단독 eval | P3-1 | 2시간 | Table 2 "+ADAT" | ⬜ |
| P3-3 | **SFA+ADAT fine-tuning** | P2-2 + P3-1 | **~24h** | `results/04_sfa_adat/` | ⬜ |
| P3-4 | Token efficiency 실측 | P3-1 | 6시간 | Figure 6 실 데이터 | ⬜ |
| P3-5 | SFA+ADAT eval | P3-3 | 2시간 | Table 1, 2 "+SFA+ADAT" | ⬜ |

### Phase 4: Full System (SCR)

| ID | 작업 | 의존성 | 시간 | 산출물 | 상태 |
|----|------|--------|------|--------|------|
| P4-1 | **SCR loss 구현** | P3-3 | 2시간 | loss 함수 | ⬜ |
| P4-2 | **Full fine-tuning** | P4-1 | **~24h** | `results/05_full/` | ⬜ |
| P4-3 | 6개 benchmark eval | P4-2 | 6시간 | Table 1 "Full" row | ⬜ |
| P4-4 | Compute cost 측정 | P4-2 | 1시간 | Table 4 | ⬜ |
| P4-5 | Hallucination case study | P4-2 | 1시간 | Appendix | ⬜ |

### Phase 5: 확장 + Cross-Architecture + 논문 ⭐ NEW

> **방법론 유의미성을 위한 핵심 실험 Phase**
> SFA가 InternVL뿐 아니라 다른 encoder-decoder 조합에서도 효과적임을 보여야 함

| ID | 작업 | 의존성 | 시간 | 산출물 | 상태 |
|----|------|--------|------|--------|------|
| P5-1 | 추가 benchmark (TextVQA, OCRBench, AI2D) | P4-2 | 4시간 | Table 1 나머지 | ⬜ |
| P5-2 | 외부 모델 공개 수치 조사 | - | 2시간 | Table 1 비교 rows | ⬜ |
| P5-3 | Multi-resolution 실험 | P4-2 | 4시간 | Appendix | ⬜ |
| P5-4 | Architecture diagram | - | 수동 | Figure 2 | ⬜ |
| **P5-5** | **SFA → Qwen2.5-VL (SigLIP+Qwen) 적용** | P2-1 | **~24h** | Table 1b row 2 | ⬜ |
| **P5-6** | **SFA → LLaVA-OV (CLIP+LLaMA) 적용** | P2-1 | **~24h** | Table 1b row 3 | ⬜ |
| P5-7 | Cross-architecture 결과 종합 + Figure 8 | P5-5, P5-6 | 2시간 | Fig 8 Bar chart | ⬜ |
| P5-8 | 논문 작성 | P4-3, P5-7 | - | `eccv2026submission.tex` | ⬜ |

**Cross-Architecture 실험 전략**:
1. SFA 모듈은 **vision encoder의 self-attention에만 적용** → encoder가 달라져도 QKV 구조가 있으면 적용 가능
2. SigLIP의 attention 구조: 표준 multi-head self-attention → SFA 직접 적용 가능
3. CLIP-ViT-L/14: 역시 표준 ViT attention → SFA 적용 가능
4. 각 모델별 `patch_*_with_sfa()` 함수를 별도 구현하되, `SFAAttention` 모듈 자체는 공유
5. Decoder (LLM)는 frozen으로 동일 — SFA의 효과가 vision 단에서 발생함을 입증

---

## 11. 타임라인

```
2026-02-20 [완료] Step 0~6: Baseline 분석 전체 완료
2026-02-20 [완료] Phase 1: 시각화 스크립트 구현 (Figure 1, 3, 5)
2026-02-20 [완료] Phase 2-1: SFA fine-tuning 스크립트 구현 + test passed
2026-02-20~21    Phase 2-2: SFA fine-tuning 실행 시작 (~24h) ← ⭐ 다음 단계
2026-02-22       Phase 2-2 완료 → Phase 2-3~8: SFA 후속 분석
2026-02-23       Phase 3-1: ADAT 구현
2026-02-23~24    Phase 3-3: SFA+ADAT fine-tuning (~24h)
2026-02-24       Phase 3-4,5: Token efficiency + eval
2026-02-25       Phase 4: SCR + Full fine-tuning (~24h)
2026-02-26       Phase 4-3~5: Full eval + compute + case study
2026-02-27~28    Phase 5-5,6: Cross-Architecture 실험 (Qwen2.5-VL, LLaVA-OV)  ⭐ NEW
2026-03-01~      Phase 5-8: 논문 작성
```

### Cross-Architecture 실험 중요성

```
┌─────────────────────────────────────────────────────┐
│ 논문 Story                                           │
│                                                      │
│ 1. SFA는 vision encoder의 attention에 structural    │
│    bias를 주입하는 범용 모듈이다                       │
│                                                      │
│ 2. InternViT에서 효과 확인 (Primary)                  │
│                                                      │
│ 3. SigLIP, CLIP 등 다른 encoder에서도 동일 효과 확인   │
│    → architecture-agnostic contribution               │
│                                                      │
│ 4. Decoder (Qwen, LLaMA, InternLM) 변경에도 robust    │
│    → encoder-side improvement가 decoder에 전이됨 입증  │
│                                                      │
│ ⇒ "SFA는 ViT attention의 근본적 한계를 해결하는        │
│    범용 모듈" 이라는 주장이 가능                        │
└─────────────────────────────────────────────────────┘
```
