# DSA Project Progress Report

**Project**: Structure-Factorized Attention (SFA) for Document-Centric Multimodal LLMs
**Last Updated**: 2026-02-23

---

## Overview

문서/차트 특화 멀티모달 태스크에서 기존 ViT의 구조적 한계를 해결하기 위해
**Structure-Factorized Attention (SFA)** + **Adaptive Density-Aware Tokenization (ADAT)** 을 제안하고
InternVL3.5-8B 위에서 검증하는 연구 프로젝트.

### Core Modules
| Module | Description | Params |
|--------|-------------|--------|
| **SFA** | Attention에 structural bias (row/col/block) 주입 | 304/layer × 24 = **7,296** |
| **ADAT** | 텍스트 밀집도 기반 동적 패치 할당 | **186K** (density estimator) |
| **SCR** | Entropy/Grounding/Stability regularization | Loss only (추가 params 없음) |

### Environment
| Item | Value |
|------|-------|
| Base Model | InternVL3.5-8B (InternViT-300M + InternLM2.5-7B) |
| GPU | NVIDIA A100-PCIE-40GB × 2 |
| Framework | PyTorch 2.4.0, transformers 5.1.0 |
| Conda env | `docmllm` |

---

## Completed Steps

### Step 0: Baseline Evaluation ✅

InternVL3.5-8B 원본 모델의 ChartQA 성능 측정.

| Benchmark | Metric | Score | Samples |
|-----------|--------|-------|---------|
| ChartQA | Relaxed Accuracy | **0.6200** | 200 |

- Single-tile (448×448) 추론
- 오답 패턴: 소수점 수치 오류 (0.57→10.04), 미세 차이 (0.08→0.02)
- **숫자 grounding 취약점 확인**

### Step 1: Text Density Estimator ✅

문서 이미지의 텍스트 밀집도를 예측하는 lightweight CNN 학습.

| Item | Value |
|------|-------|
| Architecture | 6-layer CNN (3→32→64→128→64→32→1) |
| Parameters | **186K** |
| Train/Val | 19,000 / 1,000 (ChartQA + DVQA) |
| Best Val Loss | **0.001728** (Epoch 10) |
| Output | 28×28 density heatmap D(x,y) ∈ [0,1] |

Pseudo label 생성: Canny edge → adaptive threshold → Gaussian blur

### Step 2: SFA Module Test ✅

Structure-Factorized Attention 모듈 단독 동작 검증.

**SFA 수식:**
```
S_ij = (Q_i · K_j^T) / √d + φ(s_i, s_j)

φ(s_i, s_j) = w_row·𝟙[row_i = row_j]
            + w_col·𝟙[col_i = col_j]
            + w_dist·(-manhattan(i,j))
            + block_embed(b_i)^T · block_embed(b_j)
```

| Item | Value |
|------|-------|
| Forward test | [2, 784, 1024] → [2, 784, 1024] ✅ |
| Structural Bias Params | **304** per layer (0.007% overhead) |

### Step 3: SFA → InternVL Integration ✅

InternViT의 24개 self-attention layer를 SFA로 교체 후 inference 확인.

| Item | Value |
|------|-------|
| Replaced layers | **24 / 24** |
| Trainable params | 337,590,400 / 8,528,325,760 (**4.0%**) |
| Inference test | **PASSED** |

### Step 4: Attention Entropy Analysis (Baseline) ✅

Text-dense vs sparse region의 attention entropy 측정.

| Region | Entropy |
|--------|---------|
| Text-dense | **4.3322** |
| Sparse | **4.4377** |
| Ratio | **0.98x** |

→ 구조적 바이어스 부재 확인 — text/sparse 간 entropy 차이 거의 없음

### Step 5: Hallucination Rate Analysis (Baseline) ✅

| Metric | Value |
|--------|-------|
| Accuracy | **0.6500** |
| Hallucination Rate | **0.2550** (51/200) |
| Wrong Answer Rate | 0.0950 (19/200) |

→ **오류의 73%가 숫자 hallucination** — 구조적 grounding 부재가 주원인

### Step 6: Token Efficiency Curve ✅ (Placeholder)

Placeholder 생성 완료. ADAT 구현 후 실 데이터로 교체 예정.

---

## In Progress

### Phase 2-2: SFA Fine-tuning 🔄 (현재 학습 중)

**OOM 문제 해결 후 학습 진행 중.**

#### OOM 해결 방법
기존 문제: 8.5B 모델을 A100-40GB에 올리면 OOM 발생 (모델 17GB + optimizer + activations)

해결:
1. **Frozen LLM → 4-bit NF4 양자화** (bitsandbytes): ~14GB → ~3.5GB
2. **Vision encoder gradient checkpointing**: 활성화 메모리 절감
3. **batch_size=1, grad_accum=32**: 피크 메모리 최소화

결과: **GPU 메모리 8.3GB / 40GB** (이전 OOM → 충분한 여유)

#### 학습 설정
| Item | Value |
|------|-------|
| Data | ChartQA train (28,299 samples) |
| Effective batch size | 1 × 32 (grad_accum) = **32** |
| Epochs | 3 |
| Total optimizer steps | 2,653 |
| LR | 2e-5 (cosine, warmup 100 steps) |
| Trainable | Vision encoder (SFA) + Projector (**337M / 4.7B = 7.1%**) |
| Frozen | LLM (4-bit quantized) |

#### 학습 경과
| Epoch | Step | Loss | LR | GPU Mem |
|-------|------|------|----|---------|
| 1 | 320/28299 | 5.5692 | 2.00e-06 | 8.3GB |
| 1 | 640/28299 | 5.4709 | 4.00e-06 | 8.3GB |

예상 학습 시간: ~12시간

---

## Generated Figures

| Figure | File | Status |
|--------|------|--------|
| Fig 1: Motivation (uniform vs adaptive patching) | `figures/fig1_motivation/` | ✅ 생성 완료 |
| Fig 2: Architecture diagram | `figures/fig2_architecture.{pdf,png}` | ✅ 생성 완료 |
| Fig 4: Density map gallery | `results/01_density/visualizations/` | ✅ 20장 생성 |
| Fig 5: Entropy (baseline) | `figures/fig5_entropy/` | ✅ Baseline 생성 (SFA 후 완성 예정) |
| Fig 6: Token efficiency | `results/04_analysis/token_efficiency_curve.{pdf,png}` | ✅ Placeholder |

---

## Remaining Phases

### Phase 2 (SFA 후속 — 학습 완료 후)
- P2-3: SFA 모델 ChartQA eval → Table 1 "+SFA"
- P2-4: SFA entropy 재측정 → Figure 5 완성
- P2-5: SFA hallucination 재측정 → Table 2 "+SFA"
- P2-6: SFA attention heatmap → Figure 3 완성
- P2-7: Structural bias 시각화 → Figure 7
- P2-8: Structural component ablation → Table 3

### Phase 3 (ADAT)
- ADAT 모듈 구현 + 단독 eval
- SFA+ADAT 통합 fine-tuning
- Token efficiency 실측

### Phase 4 (Full System + SCR)
- SCR loss 구현 + fine-tuning
- 6개 benchmark 전체 eval
- Compute cost 측정

### Phase 5 (Cross-Architecture + 논문)
- SFA → Qwen2.5-VL (SigLIP+Qwen) 적용
- SFA → LLaVA-OV (CLIP+LLaMA) 적용
- 논문 작성

---

## Technical Issues Resolved

| Issue | Cause | Solution |
|-------|-------|----------|
| `conversation.py` 누락 | `Model_original` 디렉토리 | 경로를 `/NetDisk/j_son/internvl_35/`로 변경 |
| Meta tensor RuntimeError | `device_map="auto"` + InternViT | `from_config()` + safetensors 수동 로딩 |
| Flash attn → attention weight 미노출 | InternViT 기본값 | `use_flash_attn=False` + QKV hook |
| VL loss gradient 끊김 | `img_context_token_id` 미설정 | `compute_vl_loss()` 수동 구현 |
| **CUDA OOM (학습 불가)** | 8.5B 모델 on A100-40GB | **LLM 4-bit 양자화 + gradient checkpointing** |
| Patch grid 28 vs 32 | 448/14 = 32 | 전체 모듈 `num_patches_h/w` 32로 수정 |

---

## File Structure

```
DSA/
├── plan.md                              # 1개월 연구 계획서
├── PROGRESS.md                          # 진행 현황 (이 파일)
├── architecture_diagram_prompt.md       # Figure 2 AI 생성 프롬프트
├── eccv2016submission.tex               # 논문 템플릿
├── Structure-Factorized_Document_Attention.pdf  # 참고 논문
├── experiments/
│   ├── EXP-20260220-001-experiment-design.md    # 실험 마스터 문서
│   ├── scripts/
│   │   ├── model_utils.py               # 모델 로딩 (full/quantized)
│   │   ├── 00_baseline_eval.py          # Baseline 평가
│   │   ├── 01_density_estimator.py      # Density Estimator 학습
│   │   ├── 02_sfa_module.py             # SFA 모듈 정의
│   │   ├── 03_sfa_integration.py        # SFA → InternVL 통합
│   │   ├── 03_sfa_finetune.py           # SFA Fine-tuning (4-bit quantized)
│   │   ├── 04_attention_analysis.py     # Entropy/Hallucination 분석
│   │   ├── 05_figure_motivation.py      # Figure 1 생성
│   │   ├── 06_attention_heatmap.py      # Figure 3 생성
│   │   ├── 07_figure_entropy.py         # Figure 5 생성
│   │   └── gen_architecture_diagram.py  # Figure 2 생성
│   ├── figures/
│   │   ├── fig1_motivation/             # Figure 1: Motivation
│   │   ├── fig2_architecture.{pdf,png}  # Figure 2: Architecture
│   │   ├── fig5_entropy/                # Figure 5: Entropy
│   │   └── sample_images/              # 샘플 이미지
│   └── results/
│       ├── 00_baseline/                 # Baseline 결과
│       ├── 01_density/                  # Density Estimator 체크포인트
│       ├── 03_sfa_train/                # SFA 학습 (진행 중)
│       └── 04_analysis/                 # Entropy/Hallucination 분석
└── .gitignore
```
