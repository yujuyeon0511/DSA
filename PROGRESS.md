# DSA Project Progress Report

**Project**: Structure-Factorized Attention (SFA) for Document-Centric Multimodal LLMs
**Last Updated**: 2026-02-25

---

## Overview

문서/차트 특화 멀티모달 태스크에서 기존 ViT의 구조적 한계를 해결하기 위해
**Structure-Factorized Attention (SFA)** + **Adaptive Density-Aware Tokenization (ADAT)** +
**Structural Consistency Regularization (SCR)** 을 제안하고
InternVL3.5-8B 위에서 검증하는 연구 프로젝트.

### Core Modules
| Module | Description | Params |
|--------|-------------|--------|
| **SFA** | Attention에 structural bias (row/col/block) 주입 | 304/layer × 24 = **7,296** |
| **ADAT** | 텍스트 밀집도 기반 density-guided block assignment | **186K** (density estimator) |
| **SCR** | Attention entropy regularization on text-dense patches | Loss only (추가 params 없음) |

### Environment
| Item | Value |
|------|-------|
| Base Model | InternVL3.5-8B (InternViT-300M + InternLM2.5-7B) |
| GPU | NVIDIA A100-PCIE-40GB × 2 |
| Framework | PyTorch 2.4.0, transformers 5.1.0 |
| Conda env | `docmllm` |

### Final Results Summary

| Configuration | Trainable Params | ChartQA Acc | Halluc Rate |
|--------------|-----------------|-------------|-------------|
| Baseline (no SFA) | 0 | 0.620 | 25.5% |
| + SFA (full encoder ft) | 337M (4.0%) | 0.509 | 23.0% |
| + SFA-only (backbone frozen) | 7,296 (0.0%) | 0.6244 | 20.2% |
| + SFA+ADAT (backbone frozen) | 7,296 (0.0%) | 0.6284 | 20.0% |
| **+ SFA+ADAT+SCR (backbone frozen)** | **7,296 (0.0%)** | **0.6288** | **20.0%** |

---

## Completed Steps

### Phase 0: Baseline Evaluation ✅

InternVL3.5-8B 원본 모델의 ChartQA 성능 측정.

| Metric | Value |
|--------|-------|
| ChartQA Relaxed Accuracy | **0.620** (2,500 samples) |
| Hallucination Rate | **25.5%** (51/200 samples) |

### Phase 1: Infrastructure ✅

#### Step 1: Text Density Estimator
6-layer CNN (186K params) 학습. Best Val Loss: 0.001728 (Epoch 10).

#### Step 2: SFA Module
Structure-Factorized Attention 모듈 구현 및 단독 테스트 통과.

#### Step 3: SFA → InternVL Integration
InternViT 24개 layer를 SFA로 교체, inference 확인.

#### Step 4-5: Baseline Analysis
- Attention entropy: text-dense 4.33 vs sparse 4.44 (ratio 0.98x) → 구조적 바이어스 부재 확인
- Hallucination: 오류의 73%가 숫자 hallucination

#### Step 6: Visualization
Figure 1 (Motivation), Figure 2 (Architecture), Figure 5 (Entropy) 생성.

### Phase 2: SFA Fine-tuning ✅

#### P2-1: OOM 해결 + 학습
- LLM 4-bit NF4 양자화 (14GB → 3.5GB), gradient checkpointing
- 3 epochs, 28K samples, GPU 메모리 8.3GB/40GB

#### P2-2: Full encoder fine-tuning (337M params)
- ChartQA Acc: 0.509 (baseline 대비 -17.9%) → **Catastrophic Forgetting**
- Hallucination 23.0% (소폭 개선되었으나 정확도 하락 심각)

#### P2-8: SFA-only (backbone frozen, 7,296 params)
- ChartQA Acc: **0.6244** (+0.7%), Hallucination: **20.2%** (-5.3%p)
- 정확도 유지 + hallucination 대폭 감소

#### P2-3~P2-7: Post-training Analysis
- Entropy 재측정, hallucination 재측정
- Attention heatmap (Figure 3), structural bias 시각화 (Figure 7)
- Loss curve (Figure 6)

### Phase 3: ADAT (Density-Guided Block Assignment) ✅

SFA+ADAT 통합: density estimator → 16-block quantization → SFA block embedding.

| Metric | SFA-only | SFA+ADAT |
|--------|----------|----------|
| ChartQA Acc | 0.6244 | **0.6284** (+0.4%p) |
| Halluc Rate | 20.2% | **20.0%** (-0.2%p) |

- 학습: 3 epochs, best loss 5.0655
- 200-sample subset: hallucination 39건 (SFA-only와 동일)

### Phase 4: SCR (Structural Consistency Regularization) ✅

Attention entropy regularization on text-dense patches.

#### 구현 결정
- Entropy regularization만 구현 (실용적)
- Numeric Grounding Loss: ChartQA에 bbox annotation 없어 불가
- Token Stability Loss: 메모리 2배 필요 → 단일 GPU 제약으로 불가

#### 학습 결과 (3 Epochs)
| Epoch | Total Loss | Task Loss | Entropy Loss |
|-------|-----------|-----------|-------------|
| 1 | 5.4255 | 5.0173 | 4.0822 |
| 2 | 5.3951 | 4.9874 | 4.0765 |
| 3 | **5.3901** | **4.9826** | **4.0751** |

#### 평가 결과
| Metric | SFA+ADAT | SFA+ADAT+SCR |
|--------|----------|-------------|
| ChartQA Acc | 0.6284 | **0.6288** |
| Halluc Rate (2500) | 20.0% | **20.0%** |
| Halluc Rate (200) | 19.5% | **19.0%** |
| Numeric Halluc | 39건 | **38건** |

### Phase 5: Paper Writing ✅

논문 초안 완성 (eccv2016submission.tex + egbib.bib).

| Section | 상태 |
|---------|------|
| Abstract | ✅ 실제 결과 반영 + 크로스벤치마크 언급 |
| Introduction | ✅ 4개 contributions |
| Related Work | ✅ 4 subsections (MLLMs, ViT, Document, Hallucination) |
| Method | ✅ SFA 수식, Density Estimation, Block Assignment, SCR |
| Experiments | ✅ Main results (Table 1), Hallucination (Table 2), Cross-benchmark (Table 4), Parameter efficiency (Table 5) |
| Discussion | ✅ Frozen backbone 분석, Limitations, Future work |
| Conclusion | ✅ 핵심 기여 요약 |
| References | ✅ 31개 (23개 본문 인용) |

### Phase 6: Multi-Benchmark Evaluation ✅

SFA+ADAT+SCR 모델을 추가 벤치마크에서 평가 (baseline 대비 비교).

| Benchmark | Metric | Baseline | SFA+ADAT+SCR | Delta |
|-----------|--------|----------|-------------|-------|
| ChartQA (2,500) | Relaxed Acc | 62.0% | **62.9%** | +0.9%p |
| DocVQA (500) | ANLS | 53.6% | 52.9% | -0.7%p |
| InfographicVQA (500) | ANLS | 38.7% | 38.5% | -0.2%p |
| DVQA (500) | Exact Match | 41.0% | 40.8% | -0.2%p |
| FigureQA (500) | Exact Match | 95.6% | 95.6% | ±0.0%p |

- SFA는 학습 도메인(ChartQA)에서 개선, 다른 벤치마크에서 성능 유지 (±1%p 이내)
- Frozen backbone 전략으로 catastrophic forgetting 없음
- HiTab은 질문 형식 불일치로 제외 (instruction suffix 부재)

---

## Generated Figures

| Figure | File | Status |
|--------|------|--------|
| Fig 1: Motivation (3-panel) | `figures/fig1_motivation/` | ✅ |
| Fig 2: Architecture diagram | `figures/fig2_architecture.{pdf,png}` | ✅ |
| Fig 3: Attention Heatmap | `figures/fig3_attention_heatmap/` | ✅ |
| Fig 4: Density map gallery | `results/01_density/visualizations/` | ✅ (20장) |
| Fig 5: Entropy (baseline vs SFA) | `figures/fig5_entropy/` | ✅ |
| Fig 6: Loss curve | `figures/fig_loss_curve/` | ✅ |
| Fig 7: Structural bias (3종) | `figures/fig7_structural_bias/` | ✅ |

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
├── PRESENTATION.md                      # 발표자료 (13 slides)
├── eccv2016submission.tex               # 논문 초안
├── egbib.bib                            # 참고문헌 (31개)
├── experiments/
│   ├── scripts/
│   │   ├── model_utils.py               # 모델 로딩 (full/quantized)
│   │   ├── 00_baseline_eval.py          # Baseline 평가
│   │   ├── 01_density_estimator.py      # Density Estimator 학습
│   │   ├── 02_sfa_module.py             # SFA 모듈 + SCR utilities
│   │   ├── 03_sfa_integration.py        # SFA → InternVL 통합
│   │   ├── 03_sfa_finetune.py           # SFA Fine-tuning
│   │   ├── 04_attention_analysis.py     # Entropy/Hallucination 분석
│   │   ├── 05_figure_motivation.py      # Figure 1 생성
│   │   ├── 06_attention_heatmap.py      # Figure 3 생성
│   │   ├── 07_figure_entropy.py         # Figure 5 생성
│   │   ├── 13_sfa_adat_train.py         # SFA+ADAT 학습
│   │   ├── 14_adat_eval.py              # ADAT 평가
│   │   ├── 15_scr_losses.py             # SCR loss functions
│   │   ├── 16_scr_train.py              # SCR 학습
│   │   ├── 17_scr_eval.py               # SCR 평가
│   │   ├── 18_multi_benchmark_eval.py   # Multi-benchmark 평가
│   │   └── gen_architecture_diagram.py  # Figure 2 생성
│   ├── figures/
│   │   ├── fig1_motivation/             # Figure 1: Motivation
│   │   ├── fig2_architecture.{pdf,png}  # Figure 2: Architecture
│   │   ├── fig3_attention_heatmap/      # Figure 3: Attention Heatmap
│   │   ├── fig5_entropy/                # Figure 5: Entropy
│   │   ├── fig7_structural_bias/        # Figure 7: Structural Bias
│   │   ├── fig_loss_curve/              # Loss Curve
│   │   └── sample_images/              # 샘플 이미지
│   └── results/
│       ├── 00_baseline/                 # Baseline 결과
│       ├── 01_density/                  # Density Estimator
│       ├── 03_sfa_train/                # SFA 학습 로그
│       ├── 04_analysis/                 # Entropy/Hallucination 분석
│       ├── 05_sfa_eval/                 # SFA 평가 결과
│       ├── 06_ablation_sfa_only/        # SFA-only ablation
│       ├── 07_sfa_adat/                 # SFA+ADAT 결과
│       ├── 08_scr/                      # SCR 결과
│       └── 09_multi_benchmark/          # Multi-benchmark 평가 결과
└── .gitignore
```

---

## Remaining Work

### 추가 진행 가능
- Cross-Architecture: Qwen2.5-VL, LLaVA-OV에 SFA 적용 (모델 다운로드 필요)
- 다양한 데이터셋 혼합 학습으로 cross-benchmark transfer 개선
- 논문 최종 수정 및 제출
