# Structure-Factorized Attention (SFA) for Document-Centric Multimodal LLMs

## 발표자료

---

## Slide 1. 연구 배경 및 문제 정의

### 기존 Vision Encoder의 한계

현재 대부분의 멀티모달 LLM (InternVL, Qwen-VL, LLaVA 등)은 **ViT 기반 vision encoder**를 사용.

**문제점:**

- **Uniform Patch Tokenization**: 문서 이미지에서 텍스트 밀집 영역과 빈 공간에 동일한 크기의 패치 적용
  - 텍스트 영역: 정보가 과도하게 압축 → 작은 숫자/텍스트 인식 오류
  - 빈 공간: 불필요한 토큰 낭비

- **Layout Inductive Bias 부재**: ViT의 positional encoding은 문서의 행/열/블록 구조를 반영하지 못함

- **숫자 Hallucination**: 차트/표에서 존재하지 않는 숫자를 생성하는 문제가 심각

### 핵심 연구 질문

> 문서 이미지의 **구조적 특성**을 vision encoder의 attention에 직접 주입하면,
> grounding이 안정화되고 hallucination이 줄어드는가?

#### 📎 이 슬라이드에 포함할 자료

| 자료 | 파일 경로 | 상태 |
|------|----------|------|
| **Figure 1: Motivation** — (a) Uniform grid (b) Density heatmap (c) Adaptive patching | `experiments/figures/fig1_motivation/figure1_motivation.png` | ✅ 완료 |
| **ChartQA 샘플 이미지** — 문제 상황 예시용 차트 이미지 | `experiments/figures/sample_images/chartqa_sample.png` | ✅ 완료 |

---

## Slide 2. Baseline 분석 결과

### 실험 환경

| 항목 | 값 |
|------|-----|
| Base Model | InternVL3.5-8B |
| Vision Encoder | InternViT-300M-448px (24 layers) |
| LLM | InternLM2.5-7B-Chat |
| GPU | NVIDIA A100-PCIE-40GB × 2 |
| 평가 벤치마크 | ChartQA (Relaxed Accuracy) |

### Baseline 성능

| Metric | Score |
|--------|-------|
| **ChartQA Relaxed Accuracy** | **0.620** |

### Hallucination 분석 (200 samples)

| 분류 | 수 | 비율 |
|------|-----|------|
| 정답 | 130 | 65.0% |
| **숫자 Hallucination** | **51** | **25.5%** |
| 오답 (기타) | 19 | 9.5% |

> **오류의 73%가 숫자 hallucination** — 이미지에 없는 숫자를 생성

### 오답 예시

| Question | GT | Prediction | 오류 유형 |
|----------|-----|-----------|-----------|
| "What was the largest dark red bar value?" | 0.08 | **26** | 숫자 hallucination |
| "What is the difference between highest and lowest?" | 54 | **99** | 숫자 hallucination |
| "What's the value for the rightmost bar?" | 0.57 | **10.04** | 자릿수 오류 |

#### 📎 이 슬라이드에 포함할 자료

| 자료 | 파일 경로 | 상태 |
|------|----------|------|
| **Table: Baseline 성능** — ChartQA Relaxed Accuracy = 0.620 | `experiments/results/00_baseline/summary.json` | ✅ 완료 |
| **Table: Hallucination 분석** — 200 samples 분류 (정답/halluc/오답) | `experiments/results/04_analysis/hallucination_analysis.json` | ✅ 완료 |
| **Table: 오답 예시** — hallucination_analysis.json에서 대표 오답 3건 추출 | `experiments/results/04_analysis/hallucination_analysis.json` | ✅ 완료 |

---

## Slide 3. Attention Entropy 분석 (Baseline)

### Attention Entropy 분석

| 영역 | Entropy | 차이 |
|------|---------|------|
| Text-dense region | 4.3322 | - |
| Sparse region | 4.4377 | - |
| **비율** | **0.98x** | ≈ 동일 |

> Text-dense/sparse 간 attention entropy 차이가 거의 없음
> → **Vision encoder가 문서 구조를 구분하지 못함** → SFA 필요성 입증

#### 📎 이 슬라이드에 포함할 자료

| 자료 | 파일 경로 | 상태 |
|------|----------|------|
| **Figure 5: Entropy Analysis (Baseline)** — Violin plot + Layer-wise line plot | `experiments/figures/fig5_entropy/fig5_entropy.png` | ✅ 완료 |
| **Table: Entropy 통계** — text-dense vs sparse 영역 entropy 비교 | `experiments/results/04_analysis/entropy_analysis.json` | ✅ 완료 |
| **Figure 5: Entropy Analysis (Baseline vs SFA)** — 학습 후 비교 추가 | ⏳ SFA 학습 완료 후 재생성 | ⏳ 예정 (P2-4) |

---

## Slide 4. 제안 방법: Structure-Factorized Attention (SFA)

### 핵심 아이디어

기존 ViT self-attention에 **문서 구조 bias**를 추가:

```
기존:  S_ij = (Q_i · K_j^T) / √d

제안:  S_ij = (Q_i · K_j^T) / √d  +  φ(s_i, s_j)
              ─────────────────     ──────────────
               Content Attention    Structural Bias
```

### Structural Bias φ(s_i, s_j) 구성

```
φ = w_row · 𝟙[row_i = row_j]           ← 같은 행 패치 간 강화
  + w_col · 𝟙[col_i = col_j]           ← 같은 열 패치 간 강화
  + w_dist · (-manhattan(i,j))          ← 가까운 패치 간 강화
  + block_embed(b_i)^T · block_embed(b_j)  ← 같은 블록 간 강화
```

### 파라미터 효율성

| Component | Parameters | Overhead |
|-----------|-----------|----------|
| w_row (per head) | 16 | - |
| w_col (per head) | 16 | - |
| w_dist (per head) | 16 | - |
| block_embed (16 blocks × 16 heads) | 256 | - |
| **Layer당 합계** | **304** | **0.007%** |
| **전체 (24 layers)** | **7,296** | **0.002%** |

> 전체 모델 대비 **0.002%의 파라미터**만 추가하면서 구조적 inductive bias 제공

### 적용 방식

- InternViT의 **24개 attention layer 모두**에 SFA 적용
- CLS 토큰에는 structural bias 미적용 (spatial tokens만)
- Pretrained QKV weights 유지, structural bias만 small init (std=0.02)

#### 📎 이 슬라이드에 포함할 자료

| 자료 | 파일 경로 | 상태 |
|------|----------|------|
| **Figure 2: Architecture Diagram** — 전체 파이프라인 구조도 | `experiments/figures/fig2_architecture.png` | ✅ 완료 |
| **Table: 파라미터 효율성** — SFA 추가 파라미터 분석 (위 표 사용) | 본문 내 표 | ✅ 완료 |

---

## Slide 5. Adaptive Density-Aware Tokenization (ADAT)

### 동기

문서 이미지에서:
- 텍스트 밀집 영역 → 더 작은 패치 필요 (높은 해상도)
- 빈 공간 → 큰 패치로 충분 (토큰 절약)

### Text Density Estimator

| 항목 | 값 |
|------|-----|
| Architecture | 6-layer CNN |
| Parameters | **186K** (경량) |
| Input | 448×448 document image |
| Output | 28×28 density heatmap D(x,y) ∈ [0,1] |
| Training | Self-supervised (pseudo labels) |
| Val Loss | **0.001728** |

### Pseudo Label 생성

```
Document Image → Canny Edge Detection → Adaptive Threshold → Gaussian Blur → Density Map
```

별도 annotation 없이 이미지 자체에서 텍스트 밀도 추정 가능

### Adaptive Patch 전략 (계획)

| Density | Patch Size | 토큰 수 |
|---------|-----------|---------|
| D > 0.7 (high) | 8×8 | 많음 (세밀) |
| 0.3 < D ≤ 0.7 (medium) | 14×14 | 표준 |
| D ≤ 0.3 (low) | 32×32 | 적음 (효율) |

#### 📎 이 슬라이드에 포함할 자료

| 자료 | 파일 경로 | 상태 |
|------|----------|------|
| **Figure 1: Motivation (b) Density Heatmap** — 밀도 추정 결과 | `experiments/figures/fig1_motivation/figure1_motivation.png` (panel b) | ✅ 완료 |
| **Density Map 시각화** — 20개 ChartQA 이미지의 density estimation 결과 | `experiments/results/01_density/visualizations/density_000~019.png` | ✅ 완료 |
| **Token Efficiency Curve** — 밀도 기반 토큰 절약 효과 그래프 | `experiments/results/04_analysis/token_efficiency_curve.png` | ✅ 완료 |

---

## Slide 6. 학습 전략 및 OOM 해결

### 학습 구조

```
┌─────────────────────────────────────────────────┐
│  Trainable (4.0%)                               │
│  ┌──────────────────┐  ┌──────────────────┐     │
│  │ InternViT-300M   │  │ MLP Projector    │     │
│  │ + SFA (24 layers)│→│ (4096→4096)      │     │
│  │ GPU 0: 8.1 GB    │  │                  │     │
│  └──────────────────┘  └────────┬─────────┘     │
│                                 │               │
│  Frozen (96.0%)                 ▼               │
│  ┌──────────────────────────────────────────┐   │
│  │ InternLM2.5-7B-Chat (bf16)              │   │
│  │ GPU 1: 15.3 GB                          │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### OOM 문제 및 해결

**문제**: InternVL3.5-8B (17GB bf16)을 A100-40GB에서 fine-tuning 시 OOM

**해결 (2-GPU Model Parallel):**

| 전략 | 내용 | 효과 |
|------|------|------|
| Vision → GPU 0 | Vision encoder + projector (trainable) | 8.1 GB |
| LLM → GPU 1 | LLM (frozen, bf16 full) | 15.3 GB |
| Gradient Checkpointing | Vision encoder 활성화 메모리 절감 | -50% activation |
| batch_size=4 | GPU 여유 활용 | 처리량 4배↑ |

**이전 시도 (실패):**
- Single GPU full bf16 → OOM (17GB + optimizer + activations > 40GB)
- DDP 2-GPU → 각 rank마다 full model 복사 → OOM

#### 📎 이 슬라이드에 포함할 자료

| 자료 | 파일 경로 | 상태 |
|------|----------|------|
| **학습 구조 다이어그램** — 2-GPU 배치 (위 ASCII 다이어그램 사용) | 본문 내 다이어그램 | ✅ 완료 |
| **Table: OOM 해결 전략** — GPU 분할 + checkpointing 효과 (위 표 사용) | 본문 내 표 | ✅ 완료 |
| **Table: GPU 메모리 사용량** — 실측 값 | `experiments/results/03_sfa_train/train.log` (G0:8.1GB G1:15.3GB) | ✅ 완료 |

---

## Slide 7. SFA 학습 진행 상황 및 Loss 추이

### 학습 설정

| 항목 | 값 |
|------|-----|
| Data | ChartQA train (28,299 samples) |
| Effective batch size | 4 × 8 (grad_accum) = **32** |
| Epochs | 3 |
| Optimizer | AdamW (lr=2e-5, cosine schedule) |
| Total optimizer steps | 2,652 |

### Loss 추이

```
Epoch 1:
  step   80 | loss: 4.7813  ← 초기
  step  400 | loss: 4.3514
  step 2000 | loss: 1.3860
  step 7074 | loss: 1.1015  ← Epoch 1 완료

Epoch 2:
  step   80 | loss: 0.6696
  step 2000 | loss: 0.5361
  step 7074 | loss: 0.5088  ← Epoch 2 완료

Epoch 3:
  step   80 | loss: 0.5311
  step 2000 | loss: 0.4683
  step 7074 | loss: 0.4579  ← Epoch 3 완료 (Best)
```

> **Loss: 4.78 → 0.46 (약 90.4% 감소)** — 3 epochs 학습 완료

### GPU 활용률

| GPU | 역할 | 메모리 사용 | 여유 |
|-----|------|-----------|------|
| GPU 0 | Vision (trainable) | 8.1 GB | 32.9 GB (80%) |
| GPU 1 | LLM (frozen) | 15.3 GB | 25.7 GB (63%) |

#### 📎 이 슬라이드에 포함할 자료

| 자료 | 파일 경로 | 상태 |
|------|----------|------|
| **Loss Curve 그래프** — train.log에서 추출하여 생성 | `experiments/results/03_sfa_train/train.log` | ⏳ 그래프 생성 예정 |
| **Table: 학습 설정** — hyperparameter 요약 (위 표 사용) | 본문 내 표 | ✅ 완료 |
| **Table: GPU 활용률** — 실측 GPU 메모리 (위 표 사용) | 본문 내 표 | ✅ 완료 |
| **학습 로그 데이터** — step별 loss/lr 기록 | `experiments/results/03_sfa_train/train_log.json` | ✅ 완료 |

---

## Slide 8. SFA 성능 평가 결과

### Table 1: ChartQA 성능 비교 (Main Result)

| Model | ChartQA Relaxed Acc | 변화 |
|-------|-------------------|------|
| InternVL3.5-8B (Baseline) | 0.620 | - |
| **+ SFA (full encoder ft)** | **0.509** | **-0.111** |

### Table 2: Hallucination 비교 (200 samples)

| 분류 | Baseline | + SFA | 변화 |
|------|----------|-------|------|
| 정답 | 130 (65.0%) | 105 (52.5%) | -12.5%p |
| 숫자 Hallucination | 51 (25.5%) | 46 (23.0%) | **-2.5%p** |
| 오답 (기타) | 19 (9.5%) | 49 (24.5%) | +15.0%p |

### 분석: Catastrophic Forgetting 문제

> **전체 vision encoder (300M params)를 28K ChartQA만으로 fine-tuning하여 성능 하락 발생**
>
> - Hallucination rate는 소폭 개선 (25.5% → 23.0%)
> - 그러나 전체 정확도 하락 (0.620 → 0.509)이 심각
> - 원인: pretrained 시각적 이해 능력의 catastrophic forgetting
>
> **다음 실험 방향:**
> - SFA structural bias만 학습 (7,296 params), vision encoder backbone freeze
> - 또는 더 다양한 학습 데이터 사용 (ChartQA + DocVQA + 기타)

#### 📎 이 슬라이드에 포함할 자료

| 자료 | 파일 경로 | 상태 |
|------|----------|------|
| **Table 1: ChartQA 성능 비교** — Baseline vs +SFA | `experiments/results/05_sfa_eval/eval_results.json` | ✅ 완료 |
| **Table 2: Hallucination 비교** — Baseline vs +SFA 200-sample 분류 | `experiments/results/05_sfa_eval/hallucination_sfa.json` | ✅ 완료 |

---

## Slide 9. Entropy 분석 — Baseline vs SFA

### Attention Entropy 비교

| 영역 | Baseline | + SFA | 변화 |
|------|----------|-------|------|
| Text-dense region | 4.3322 | 4.7397 | +0.408 |
| Sparse region | 4.4377 | 4.7447 | +0.307 |
| Dense/Sparse Delta | -0.106 | **-0.005** | **Delta 축소** |

> - SFA 적용 후 전체 entropy가 증가 (4.33 → 4.74)
> - text/sparse 간 Delta가 -0.106 → -0.005로 거의 0에 수렴
> - 이는 SFA가 구조적 차이를 균일화했으나, 전체 entropy 증가가 정보 손실을 시사
> - Vision encoder full fine-tuning으로 인한 attention 패턴 변화

#### 📎 이 슬라이드에 포함할 자료

| 자료 | 파일 경로 | 상태 |
|------|----------|------|
| **Figure 5: Entropy Analysis (Baseline vs SFA)** — Violin + Layer-wise 비교 | `experiments/figures/fig5_entropy/fig5_entropy.png` | ✅ 완료 (재생성됨) |
| **Table: Entropy 비교** — Baseline vs SFA entropy 통계 | `experiments/results/05_sfa_eval/entropy_analysis_sfa.json` | ✅ 완료 |

---

## Slide 10. Structural Bias 시각화 & Attention Heatmap

### Figure 7: 학습된 Structural Bias

- 24개 layer × 16 heads의 w_row, w_col, w_dist 값 시각화
- Layer별 bias magnitude 분석
- 마지막 layer의 structural bias matrix 시각화

### Attention Map 비교 — Baseline vs SFA

동일 차트 이미지에 대한 attention heatmap 비교 (추후 생성 예정)

#### 📎 이 슬라이드에 포함할 자료

| 자료 | 파일 경로 | 상태 |
|------|----------|------|
| **Figure 7(a): Bias Heatmap** — w_row/w_col/w_dist per layer & head | `experiments/figures/fig7_structural_bias/fig7a_bias_heatmap.png` | ✅ 완료 |
| **Figure 7(b): Bias Bar** — Layer-wise mean magnitude | `experiments/figures/fig7_structural_bias/fig7b_bias_bar.png` | ✅ 완료 |
| **Figure 7(c): Bias Matrix** — Layer 23 structural bias matrix | `experiments/figures/fig7_structural_bias/fig7c_bias_matrix.png` | ✅ 완료 |
| **Figure 3: Attention Heatmap (Baseline vs SFA)** — 동일 이미지 비교 | ⏳ 추후 생성 (06_attention_heatmap.py) | ⏳ 예정 |

---

## Slide 11. Ablation Study

### Table 3: Component Ablation

| Configuration | ChartQA Acc | Entropy Ratio | Halluc Rate |
|--------------|-------------|---------------|-------------|
| Baseline (no SFA) | 0.620 | 0.98x | 25.5% |
| + row/col only | ⏳ | ⏳ | ⏳ |
| + row/col + dist | ⏳ | ⏳ | ⏳ |
| + full SFA (all components) | ⏳ | ⏳ | ⏳ |

#### 📎 이 슬라이드에 포함할 자료

| 자료 | 파일 경로 | 상태 |
|------|----------|------|
| **Table 3: Component Ablation** — SFA 각 구성요소의 기여도 분석 | ⏳ 학습 완료 후 ablation 실험 필요 | ⏳ 예정 (P2-8) |

---

## Slide 12. 실험 파이프라인 전체 구조

```
Phase 0: Baseline ──────────────────────── ✅ 완료
  └→ ChartQA Acc=0.620, Halluc=25.5%

Phase 1: 시각화 스크립트 ────────────────── ✅ 완료
  └→ Figure 1, 2, 5 스크립트 + Baseline 생성

Phase 2: SFA Fine-tuning ───────────────── ✅ 주요 완료
  ├→ P2-1: 스크립트 구현 ✅
  ├→ P2-2: 학습 실행 ✅ (3 epochs, loss 4.78→0.46)
  ├→ P2-3: ChartQA eval ✅ (Acc: 0.509, 하락 → Catastrophic Forgetting)
  ├→ P2-4: Entropy 재측정 ✅ (Figure 5 재생성 완료)
  ├→ P2-5: Hallucination 재측정 ✅ (23.0%, 소폭 개선)
  ├→ P2-6: Attention heatmap ⏳ (추후 생성)
  ├→ P2-7: Structural bias 시각화 ✅ (Figure 7 생성)
  └→ P2-8: Component ablation ⏳ (SFA-only ft 후 비교)

Phase 3: ADAT 구현 + 통합 ─────────────── ⬜ 예정
  └→ Adaptive tokenization + Token efficiency 실측

Phase 4: Full System (SCR) ────────────── ⬜ 예정
  └→ Entropy/Grounding/Stability loss → Hallucination 감소

Phase 5: Cross-Architecture + 논문 ────── ⬜ 예정
  ├→ SFA → Qwen2.5-VL (SigLIP+Qwen) 적용
  ├→ SFA → LLaVA-OV (CLIP+LLaMA) 적용
  └→ 논문 작성
```

#### 📎 이 슬라이드에 포함할 자료

| 자료 | 파일 경로 | 상태 |
|------|----------|------|
| 파이프라인 다이어그램 (위 텍스트 구조도) | 본문 내 | ✅ 완료 |

---

## Slide 13. 기대 기여점 및 타임라인

### 기대 기여점

1. **Structure-Factorized Attention**: 0.002% 파라미터 추가로 문서 구조 인식 강화
2. **Adaptive Density-Aware Tokenization**: 동일 토큰 수에서 정보량 극대화
3. **Hallucination 감소**: Attention entropy 감소 + 숫자 grounding 안정화
4. **Architecture-Agnostic**: InternViT/SigLIP/CLIP 등 다양한 ViT에 적용 가능
5. **다국어 일반화**: 한국어(AIDA, AIHUB) + 영어(ChartQA, DocVQA) 동시 검증

### 타임라인

| 기간 | 작업 | 상태 |
|------|------|------|
| 2/20 | Baseline 분석 + Density Estimator + SFA 모듈 | ✅ 완료 |
| 2/20 | SFA 통합 + Entropy/Hallucination 분석 | ✅ 완료 |
| 2/23 | OOM 해결 + 2-GPU 학습 시작 | ✅ 완료 |
| 2/23~24 | SFA 학습 완료 + 후속 분석 | 🔄 진행 중 |
| 2/24~25 | ADAT 구현 + SFA+ADAT 통합 | ⬜ 예정 |
| 2/25~26 | Full System (SCR) 학습 | ⬜ 예정 |
| 2/27~28 | Cross-Architecture 실험 | ⬜ 예정 |
| 3/1~ | 논문 작성 | ⬜ 예정 |

#### 📎 이 슬라이드에 포함할 자료

| 자료 | 파일 경로 | 상태 |
|------|----------|------|
| 타임라인 표 (위 표 사용) | 본문 내 표 | ✅ 완료 |

---

## 자료 현황 요약

### ✅ 완료된 Figure/Table 목록

| # | 자료명 | 파일 경로 | 사용 슬라이드 |
|---|--------|----------|-------------|
| 1 | Figure 1: Motivation (3-panel) | `experiments/figures/fig1_motivation/figure1_motivation.png` | Slide 1, 5 |
| 2 | Figure 2: Architecture Diagram | `experiments/figures/fig2_architecture.png` | Slide 4 |
| 3 | Figure 5: Entropy (Baseline vs SFA) | `experiments/figures/fig5_entropy/fig5_entropy.png` | Slide 3, 9 |
| 4 | Figure 7: Structural Bias (3종) | `experiments/figures/fig7_structural_bias/fig7{a,b,c}_*.png` | Slide 10 |
| 5 | ChartQA 샘플 이미지 | `experiments/figures/sample_images/chartqa_sample.png` | Slide 1 |
| 6 | Density Map 시각화 (20장) | `experiments/results/01_density/visualizations/density_000~019.png` | Slide 5 |
| 7 | Token Efficiency Curve | `experiments/results/04_analysis/token_efficiency_curve.png` | Slide 5 |
| 8 | Baseline 성능 | `experiments/results/00_baseline/summary.json` | Slide 2, 8 |
| 9 | Baseline Hallucination 분석 | `experiments/results/04_analysis/hallucination_analysis.json` | Slide 2 |
| 10 | Baseline Entropy 분석 | `experiments/results/04_analysis/entropy_analysis.json` | Slide 3 |
| 11 | **SFA Eval 결과** | `experiments/results/05_sfa_eval/eval_results.json` | Slide 8 |
| 12 | **SFA Hallucination 분석** | `experiments/results/05_sfa_eval/hallucination_sfa.json` | Slide 8 |
| 13 | **SFA Entropy 분석** | `experiments/results/05_sfa_eval/entropy_analysis_sfa.json` | Slide 9 |
| 14 | 학습 로그 데이터 | `experiments/results/03_sfa_train/train_log.json` | Slide 7 |

### ⏳ 추후 생성 예정

| # | 자료명 | 사용 슬라이드 | 비고 |
|---|--------|-------------|------|
| 1 | Figure 3: Attention Heatmap (Baseline vs SFA) | Slide 10 | 06_attention_heatmap.py 실행 필요 |
| 2 | Table 3: Component Ablation | Slide 11 | SFA-only training 후 생성 |
| 3 | Loss Curve 그래프 | Slide 7 | train.log 기반 생성 |

---

## Appendix: 기술적 세부사항

### SFA 수식 상세

32×32 grid (1024 spatial tokens) + 1 CLS token = 1025 tokens

Precomputed buffers (학습 중 고정):
- `same_row[i,j]`: 패치 i, j가 같은 행이면 1, 아니면 0
- `same_col[i,j]`: 같은 열이면 1
- `manhattan_dist[i,j]`: 정규화된 Manhattan 거리

```python
# 적용 위치: spatial tokens만 (CLS 제외)
attn[:, :, 1:, 1:] += φ(structural_bias)
```

### 학습 데이터 상세

| Source | Samples | 용도 |
|--------|---------|------|
| ChartQA train (augmented) | 28K | SFA fine-tuning |
| ChartQA train (human) | 7.3K | SFA fine-tuning |
| mllm_ready V3 | 5.5M rows | 추후 Phase 3-4 |
| Cauldron | 41GB | 추후 Phase 3-4 |

### 모델 구조

```
InternVL3.5-8B
├── vision_model (InternViT-300M)
│   ├── embeddings (patch_embed + pos_embed)
│   └── encoder
│       └── layers × 24
│           ├── attn → SFAAttention (교체)
│           │   ├── qkv (1024→3072)
│           │   ├── proj (1024→1024)
│           │   └── structural_bias (NEW, 304 params)
│           ├── mlp (1024→4096→1024)
│           └── layer_scale × 2
├── mlp1 (projector: 4096→4096)
└── language_model (InternLM2.5-7B, frozen)
```
