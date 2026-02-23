# Document-Specific Adaptive Vision Encoder for Multimodal LLMs

## (1-Month Fast-Track Research Plan for Top-Tier Submission)

---

# 1. 연구 목표

본 연구는 문서/차트 특화 멀티모달 태스크에서 기존 ViT 기반 vision encoder의 구조적 한계를 해결하기 위한
**Document-Specific Adaptive Vision Encoder (DSAVE)**를 제안한다.

목표는 다음과 같다:

1. 문서 특화 inductive bias를 encoder 수준에서 명시적으로 설계
2. 토큰 효율성 증가 + hallucination 감소 정량 증명
3. InternVL3.5-8B backbone을 수정하여 빠르게 검증
4. 1개월 내 실험 완료 및 탑티어 학회 제출 가능한 수준 도달

---

# 2. 기존 연구의 명확한 한계

## 2.1 Fixed Patch Tokenization의 구조적 문제

* 문서는 비균질 구조
* 텍스트 밀집 영역과 공백 영역이 극단적으로 다름
* 기존 ViT는 uniform patch 분할
* 정보 밀도 대비 토큰 비효율 발생

결과:

* 불필요한 연산 증가
* 작은 텍스트 영역 정보 손실
* 차트 축/숫자 인식 오류 발생

---

## 2.2 Layout Inductive Bias 부재

문서는:

* 2D grid
* 표 구조
* 문단 블록
* 차트 axis hierarchy

그러나 기존 모델은:

* 단순 absolute positional embedding 사용
* row/column/semantic block 정보 미반영

---

## 2.3 Hallucination 원인 가설

가설:

Hallucination은
"vision token representation collapse + language prior dominance"
에서 발생한다.

특히:

* 텍스트 영역 attention entropy 증가
* 숫자 grounding 불안정
* structural alignment 불완전

---

# 3. 제안 방법론 (구체적 설계)

본 연구는 세 가지 모듈로 구성된다.

---

# 3.1 Adaptive Density-Aware Tokenization (ADAT)

## 핵심 아이디어

문서 이미지에서 텍스트 밀집도를 예측하여
patch 크기를 동적으로 조정한다.

---

## 3.1.1 단계별 설계

### Step 1: Text Density Estimator

* lightweight CNN
* self-supervised training
* 입력: raw document image
* 출력: density heatmap D(x,y)

학습 방법:

* pseudo text mask 생성
* OCR confidence 기반 weak supervision
* edge detection + high frequency region 활용

---

### Step 2: Multi-Scale Patch Generator

Patch size:

* High density region → small patch (8x8)
* Medium density → 14x14
* Low density → 32x32

Token budget constraint 적용:

총 토큰 수 N을 고정

Optimization:

maximize:
information_gain / token_cost

---

### 기대 효과

* 동일 FLOPs 대비 정보량 증가
* 작은 숫자/텍스트 인식 향상
* token efficiency 증가

---

# 3.2 Layout-Aware Structural Embedding (LASE)

## 기존 문제

Absolute positional encoding은 구조 표현 부족

---

## 제안 방식

최종 positional embedding:

P_total = P_xy + P_row + P_col + P_block + P_semantic

---

### 구성 요소

1. Row embedding

   * 동일 y-coordinate band에 동일 embedding 부여

2. Column embedding

   * 표/차트 축 정렬 강화

3. Block embedding

   * paragraph segmentation 기반
   * connected component clustering 사용

4. Semantic region embedding

   * title / axis / legend / table header 구분

---

### 구현 방법

* clustering + lightweight segmentation
* no heavy supervision
* differentiable region encoding

---

# 3.3 Structural Consistency Regularization (SCR)

## 목표

Hallucination 감소

---

## 3.3.1 Attention Entropy Regularization

Text region attention entropy 감소 유도:

Loss_entropy = mean(Entropy(attention_text_region))

---

## 3.3.2 Numeric Grounding Loss

차트 숫자 영역 feature와 출력 숫자 alignment 강화:

contrastive loss 적용

---

## 3.3.3 Token Stability Constraint

Resolution 변경 시 representation variance 최소화:

L_stability = || f_highres - f_lowres ||

---

# 4. 실험 설계

---

## 4.1 모델 세팅

Base: InternVL3.5-8B
수정 범위: vision encoder 부분만 교체

비교군:

| # | 모델 | 설명 |
|---|------|------|
| 1 | Original InternVL3.5-8B | Baseline (InternViT-300M-448px) |
| 2 | + ADAT only | Adaptive Density-Aware Tokenization 적용 |
| 3 | + LASE only | Layout-Aware Structural Embedding 적용 |
| 4 | + ADAT + LASE | 두 모듈 동시 적용 |
| 5 | + Full (ADAT + LASE + SCR) | 전체 모듈 적용 |

추가 비교 대상 (논문 테이블용):
* InternVL2.5-8B, Qwen2.5-VL-7B, LLaVA-OneVision-7B (공개 벤치마크 수치 인용)

---

## 4.2 학습 데이터 (Training Data)

### 4.2.1 Stage 1: Text Density Estimator Pre-training

Text Density Estimator (lightweight CNN)의 self-supervised 학습에 사용.
별도 라벨 불필요 — pseudo mask를 이미지 자체에서 생성.

| 데이터 소스 | 경로 | 규모 | 용도 |
|-------------|------|------|------|
| AIHUB 교과서 텍스트 이미지 | `/NetDisk/ingyu/VLM_DATA/mllm_ready/images/AIHUB/AIHUB_subjectmaterial_text_modify/` | 506K images | 텍스트 밀집도가 높은 문서 이미지 (dense text) |
| AIHUB 교과서 이미지 | `/NetDisk/ingyu/VLM_DATA/mllm_ready/images/AIHUB/AIHUB_subjectmaterial_image_modify/` | 280K images | 그림+텍스트 혼합 문서 (mixed density) |
| ChartQA train images | `/NetDisk/juyeon/train/chartQA/ChartQA Dataset/train/png/` | 18K images | 차트 이미지 (structured layout) |
| PlotQA train images | `/NetDisk/juyeon/train/plotqa/train/png/` | ~150K images | 플롯 이미지 (axis/grid structure) |
| DVQA images | `/NetDisk/juyeon/train/dvqa/images/` | ~300K images | 바 차트 중심 (uniform structure) |
| AIDA 보고서 이미지 | `/NetDisk/ingyu/VLM_DATA/mllm_ready/images/AIDA/AIDA_report/` | 20개 도메인 | 실제 한국어 보고서 (표/차트/텍스트 혼합) |
| RVL-CDIP | `/NetDisk/juyeon/train/RVL_CDIP/images/` | 400K images | 16클래스 문서 분류 (다양한 문서 유형) |

**Density Estimator pseudo label 생성 방법:**
1. Canny edge detection → high-frequency mask
2. Adaptive thresholding → text/non-text binary mask
3. Gaussian blur → continuous density heatmap D(x,y) ∈ [0,1]
4. Optional: PaddleOCR로 추출한 텍스트 bbox confidence 활용 (weak supervision)

**학습 설정:**
* Backbone: MobileNetV3-Small (0.9M params)
* Input: 448×448 (InternVL 입력 해상도와 동일)
* Output: 28×28 density map (patch 크기 16 기준)
* Loss: MSE(predicted_density, pseudo_density)
* Epochs: 10, batch size: 256, lr: 1e-3 (cosine)
* 예상 학습 시간: ~2시간 (A100 1장)

---

### 4.2.2 Stage 2: Vision Encoder Fine-tuning (ADAT + LASE 통합)

InternVL3.5-8B의 vision encoder를 수정하여 fine-tuning.
**LLM은 freeze**, vision encoder + projector만 학습.

| 데이터 소스 | 경로 / 라벨 | 규모 | Task 유형 |
|-------------|-------------|------|-----------|
| **차트/그래프 QA** | | | |
| ChartQA (augmented) | `/NetDisk/juyeon/train/chartQA/.../train_augmented.json` | 28K QA | Chart → Answer |
| ChartQA (human) | `/NetDisk/juyeon/train/chartQA/.../train_human.json` | 7.3K QA | Chart → Answer |
| ChartRQA (cauldron) | `/NetDisk/juyeon/train/cauldron_data/chartqa/` | JSONL | Chart Reasoning |
| PlotQA V2 | `/NetDisk/juyeon/train/plotqa/train/qa_pairs_V2.json` | ~28M QA | Plot → Answer |
| DVQA | `/NetDisk/juyeon/train/dvqa/train_qa.json` | 2.3M QA | Bar Chart QA |
| AskChart | `/NetDisk/juyeon/train/askchart/instruct_ocr_aware_prompt.json` | OCR-aware | Chart + OCR |
| **문서 VQA** | | | |
| DocVQA (cauldron) | `/NetDisk/juyeon/train/cauldron_data/docvqa/` | JSONL + images | Document QA |
| InfographicVQA (cauldron) | `/NetDisk/juyeon/train/cauldron_data/infographic_vqa/` | JSONL + images | Infographic QA |
| BBox-DocVQA | `/NetDisk/juyeon/train/trustdoc/english/BBox-DocVQA/` | 32K QA (bbox) | Evidence-grounded DocQA |
| **OCR / 파싱** | | | |
| Arxiv 영문 OCR | `mllm_ready/labels/V3/Extraction_OCR/arxiv_eng_*.jsonl` | 130K rows | 학술 문서 OCR |
| Arxiv 한국어 OCR | `mllm_ready/labels/V3/Extraction_OCR/arxiv_kor_*.jsonl` | 107K rows | 한국어 문서 OCR |
| AIHUB 교과서 OCR | `mllm_ready/labels/V3/Extraction_OCR/aihub_subjectmaterial_text_*.jsonl` | 506K rows | 한국어 교과서 OCR |
| LaTeX 파싱 | `mllm_ready/labels/V3/Parsing/latex_update_eng_*.jsonl` | 234K rows | 수식 → LaTeX |
| HTML Table 파싱 | `mllm_ready/labels/V3/Parsing/aihub_subjectmaterial_image_parsing_*.jsonl` | 140K rows | 표 → HTML |
| **캡션 / 설명** | | | |
| AIDA 보고서 캡션 | `mllm_ready/labels/V3/Extraction_Caption/aida_report_caption_*.jsonl` | 3.2M rows | 보고서 이미지 설명 |
| AIHUB 인포그래픽 캡션 | `mllm_ready/labels/V3/Extraction_Caption/aihub_visualization_caption_*.jsonl` | 126K rows | 시각화 이미지 설명 |
| **추론 (Reasoning)** | | | |
| ChartRQA 1 | `mllm_ready/labels/V3/Reasoning/chartrqa_1_eng_*.jsonl` | 56K rows | 차트 복합 추론 |
| ChartRQA 2 | `mllm_ready/labels/V3/Reasoning/chartrqa_2_eng_*.jsonl` | 34K rows | 차트 복합 추론 |
| **한국어 QA** | | | |
| AIHUB 인포그래픽 QA | `mllm_ready/labels/V3/Extraction_ShortQA/aihub_visualization_*.jsonl` | 126K rows | 시각화 이미지 QA |
| AIHUB 수학 객관식 | `mllm_ready/labels/V3/Extraction_ShortQA/aihub_mathproblem_multiple_*.jsonl` | 12K rows | 수학 문제 풀이 |
| KISTI 차트 | `/NetDisk/ingyu/VLM_DATA/mllm_ready/images/KISTI/bichallava_instruct_230k_chart/` | 12K images | 차트 인스트럭션 |

**mllm_ready 데이터 포맷 (V3):**
```
labels/V3/
├── Extraction_OCR/        # 930K rows — 텍스트 추출
├── Extraction_Caption/    # 3.6M rows — 이미지 캡셔닝
├── Extraction_ShortQA/    # 152K rows — 짧은 QA
├── Parsing/               # 495K rows — 구조화 파싱
├── Reasoning/             # 90K rows  — 복합 추론
└── TextOnly/              # 231K rows — 텍스트 전용
                      합계: 5.5M rows
```

**학습 데이터 샘플링 전략:**

전체 5.5M rows + juyeon/train 데이터를 모두 사용하면 과도하므로, task별 균형 샘플링 적용:

| Task 카테고리 | 샘플 수 | 비율 | 근거 |
|--------------|---------|------|------|
| Chart/Plot QA | 100K | 25% | 핵심 타겟 도메인 (차트 숫자 인식) |
| Document VQA | 80K | 20% | 문서 레이아웃 이해 핵심 |
| OCR / Parsing | 80K | 20% | Text density estimator 검증 |
| Caption / Description | 60K | 15% | 전체 이미지 이해 능력 유지 |
| Reasoning | 40K | 10% | 복합 추론 (SCR 검증) |
| Korean-specific | 40K | 10% | 한국어 문서/차트 다국어 일반화 |
| **합계** | **400K** | **100%** | |

**학습 설정:**
* Vision encoder: InternViT-300M + DSAVE 모듈 (trainable)
* Projector: MLP (trainable)
* LLM: InternLM2.5-7B-Chat (frozen)
* Batch size: 128 (gradient accumulation 16 × micro-batch 8)
* Learning rate: 2e-5 (vision encoder), 1e-5 (projector)
* Epochs: 3
* 예상 학습 시간: ~24시간 (A100 80GB × 4)

---

### 4.2.3 Stage 3: SCR Fine-tuning (Structural Consistency Regularization)

Stage 2 체크포인트에서 시작하여 추가 fine-tuning.
SCR의 3가지 보조 loss를 추가하여 hallucination 감소 유도.

| 데이터 | 경로 | 규모 | SCR 활용 |
|--------|------|------|----------|
| ChartQA + bbox | BBox-DocVQA data | 32K | Numeric Grounding Loss — bbox로 숫자 위치 정답 제공 |
| DVQA + answer_bbox | `/NetDisk/juyeon/train/dvqa/train_qa.json` | 100K subset | answer_bbox 필드 활용 grounding |
| PlotQA (다중 해상도) | plotqa train images | 50K subset | Token Stability Constraint — 해상도별 비교 |
| ChartRQA Reasoning | mllm_ready Reasoning/ | 90K | Attention Entropy Reg — 추론 시 attention 분석 |

**SCR Loss 구성:**
```
L_total = L_task + α·L_entropy + β·L_grounding + γ·L_stability

α = 0.1  (attention entropy regularization weight)
β = 0.5  (numeric grounding loss weight)
γ = 0.2  (token stability constraint weight)
```

**학습 설정:**
* 초기화: Stage 2 best checkpoint
* Learning rate: 5e-6 (cosine decay)
* Epochs: 2
* 예상 학습 시간: ~12시간 (A100 80GB × 4)

---

## 4.3 평가 데이터 (Evaluation Benchmarks)

### 4.3.1 주요 벤치마크 (Main Results Table)

| 벤치마크 | 경로 / 출처 | 규모 | 측정 지표 | 평가 대상 |
|----------|-------------|------|-----------|-----------|
| ChartQA test | `/NetDisk/juyeon/train/chartQA/.../test/` | 1.5K | Relaxed Accuracy | 차트 수치 이해 |
| DocVQA test | trustdoc or HF download | 5.2K | ANLS | 문서 텍스트 이해 |
| InfographicVQA | cauldron_data/infographic_vqa | ~5K | ANLS | 인포그래픽 이해 |
| OCRBench | HF `echo840/ocrbench` | 1K | Accuracy | OCR 정확도 |
| AI2D test | LMUData or HF | 3K | Accuracy | 다이어그램 이해 |
| TextVQA val | HF download | 5K | Accuracy | 자연 이미지 텍스트 |

### 4.3.2 확장 벤치마크 (Appendix용)

| 벤치마크 | 경로 | 측정 지표 | 비고 |
|----------|------|-----------|------|
| CharXiv descriptive | `/NetDisk/juyeon/train/LMUData/CharXiv_descriptive_val.tsv` | Accuracy | 학술 차트 |
| CharXiv reasoning | `/NetDisk/juyeon/train/LMUData/CharXiv_reasoning_val.tsv` | Accuracy | 차트 추론 |
| MMBench | `/NetDisk/juyeon/train/LMUData/MMBench_DEV_EN.tsv` | Accuracy | 일반 멀티모달 |
| MME | `/NetDisk/juyeon/train/LMUData/MME.tsv` | Score | 지각 + 인지 |
| KRETA | `/NetDisk/juyeon/train/trustdoc/korean/KRETA/` | Accuracy | 한국어 문서 VQA |
| KOFFVQA | trustdoc/korean/KOFFVQA/ | Free-form | 한국어 자유형 VQA |
| PlotQA V2 test | `/NetDisk/juyeon/train/plotqa/test/` | Accuracy | 플롯 이해 |
| FigureQA test | `/NetDisk/juyeon/train/figureqa/` | Accuracy | 도형 이해 |

### 4.3.3 커스텀 평가 (Hallucination & Efficiency)

| 평가 항목 | 방법 | 데이터 |
|-----------|------|--------|
| Hallucination Rate | ChartQA 숫자 답변에서 GT와 비교, 존재하지 않는 값 생성 비율 | ChartQA test + PlotQA test |
| Attention Entropy | Vision encoder 마지막 layer attention map의 entropy 계산 | ChartQA test 이미지 |
| Token Efficiency | 동일 정확도 달성을 위한 최소 토큰 수 비교 | Budget sweep: N ∈ {64, 128, 256, 512, 1024} |
| FLOPs 분석 | fvcore.nn.FlopCountAnalysis 사용 | 단일 이미지 forward pass |
| Density Map Quality | IoU(predicted_mask, OCR_bbox_mask) on RVL-CDIP subset | RVL-CDIP 1K sample |

---

## 4.4 측정 지표 요약

| 지표 | 산식 / 설명 | 적용 벤치마크 |
|------|-------------|---------------|
| Relaxed Accuracy | ±5% tolerance 내 정답 | ChartQA |
| ANLS | Average Normalized Levenshtein Similarity | DocVQA, InfographicVQA |
| Accuracy | Exact match (대소문자 무시) | AI2D, OCRBench, TextVQA |
| Hallucination Rate (HR) | #(생성된 숫자 ∉ 이미지 내 숫자) / #(전체 숫자 답변) | ChartQA, PlotQA |
| Attention Entropy (AE) | H(α) = -Σ α_i log α_i, text region 평균 | 전체 |
| Token Efficiency (TE) | Accuracy@N / N, N = token budget | 전체 |
| FLOPs Ratio | FLOPs_DSAVE / FLOPs_baseline | 전체 |

---

# 5. 1개월 타임라인

---

## Week 1 (Day 1-7) — Baseline 재현 + Density Module

**Day 1-2: 환경 설정 + Baseline**
* InternVL3.5-8B 로드 및 추론 파이프라인 구축
* ChartQA / DocVQA / OCRBench baseline 수치 재현
* 기존 vision encoder의 attention map 시각화

**Day 3-5: Text Density Estimator 구현**
* Pseudo label 생성 파이프라인:
  - AIHUB 교과서 이미지 (`/NetDisk/ingyu/VLM_DATA/mllm_ready/images/AIHUB/`) → edge detection → density map
  - ChartQA images → density map
  - RVL-CDIP 다양한 문서 유형 → density map
* MobileNetV3-Small density predictor 학습 (2시간)
* Density heatmap 시각화 및 정성적 분석

**Day 6-7: 분석 + 논문 Introduction 초안**
* Token distribution 통계: uniform vs density-aware patch 비교
* Information density per patch 분석
* Dense text region에서의 기존 모델 failure case 수집

Deliverable:
* Density estimator 체크포인트
* Density heatmap 시각화 (Figure 2 후보)
* Token distribution 비교 그래프 (Figure 3 후보)

---

## Week 2 (Day 8-14) — Adaptive Tokenization Integration

**Day 8-10: ADAT 모듈 구현**
* Multi-scale patch generator: 8×8 / 14×14 / 32×32
* Token budget constraint: ILP solver 또는 greedy allocation
* InternViT에 ADAT 모듈 삽입 (patch embedding layer 교체)
* FLOPs 측정 (fvcore)

**Day 11-12: Stage 2 학습 (ADAT only)**
* 학습 데이터: Chart QA 100K + Doc VQA 80K + OCR 80K (Stage 2 데이터에서 subset)
* A100 × 4, ~12시간
* ChartQA / DocVQA 1차 평가

**Day 13-14: Token Efficiency 실험**
* Token budget sweep: N ∈ {64, 128, 256, 512, 1024}
* Accuracy vs Token budget 그래프 생성
* Baseline 대비 FLOPs 절감 비율 측정

Deliverable:
* ADAT 모듈 코드
* Token efficiency 비교 그래프 (Figure 4 후보)
* Ablation: ADAT only vs baseline 테이블

---

## Week 3 (Day 15-21) — Layout Embedding + SCR

**Day 15-17: LASE 모듈 구현**
* Row/Column embedding: y/x coordinate binning (28 bins)
* Block embedding: connected component 기반 clustering
* Semantic region embedding: 4-class 분류기 (title/axis/legend/body)
* P_total = P_xy + P_row + P_col + P_block + P_semantic

**Day 18-19: Stage 2 재학습 (ADAT + LASE)**
* 전체 400K 데이터로 학습
* A100 × 4, ~24시간

**Day 20-21: SCR Stage 3 학습 + Hallucination 분석**
* BBox-DocVQA bbox 활용 Numeric Grounding Loss
* DVQA answer_bbox 활용 추가 grounding
* ChartQA test에서 hallucination rate 측정
* Attention entropy 비교 (baseline vs ADAT vs ADAT+LASE vs Full)

Deliverable:
* LASE + SCR 모듈 코드
* Attention entropy 감소 결과 (Figure 5 후보)
* Hallucination rate 비교 테이블

---

## Week 4 (Day 22-30) — Full Ablation + Paper Writing

**Day 22-24: 전체 벤치마크 평가**
* Main table: ChartQA, DocVQA, InfographicVQA, OCRBench, AI2D, TextVQA
* Extended: CharXiv, MMBench, MME, KRETA, PlotQA, FigureQA
* 5-way ablation 테이블 완성

**Day 25-26: Scaling + 분석 실험**
* Resolution scaling: 224 / 448 / 896 / 1344
* Token budget scaling: 64 → 2048
* Scaling law 그래프 (log-linear fit)
* 한국어 평가 (KRETA, KOFFVQA)

**Day 27-30: 논문 작성**
* Abstract + Introduction + Related Work
* Method (ADAT, LASE, SCR 각 섹션)
* Experiments + Results
* Analysis + Ablation
* Conclusion

Deliverable:
* Full ablation table (Table 1)
* Scaling plot (Figure 6)
* 논문 초고 완성

---

# 6. 기대되는 새로운 기여점

1. **Document-specific adaptive tokenization framework**
   - 정보 밀도 기반 동적 패치 할당은 문서 AI에서 최초 시도
   - 기존 uniform patching 대비 동일 토큰 수에서 정확도 향상 정량 증명
2. **Layout-aware inductive bias formalization**
   - Row/Col/Block/Semantic 4-level positional embedding 설계
   - 표/차트 구조 인식의 명시적 임베딩 (기존: implicit only)
3. **Hallucination 감소의 구조적 원인 분석**
   - Attention entropy와 hallucination rate의 상관관계 정량 분석
   - SCR의 3가지 보조 loss가 각각 hallucination에 미치는 영향 분리
4. **Token efficiency vs accuracy trade-off 분석**
   - Budget sweep (64~2048 tokens)에서의 Pareto frontier 제시
   - DSAVE가 baseline 대비 같은 accuracy에서 ~40% 토큰 절감 목표
5. **Resolution scaling law의 새로운 관찰**
   - Adaptive tokenization에서 resolution 증가의 marginal gain 분석
   - 고해상도 이미지에서 density-aware 접근이 더 유리함을 증명
6. **다국어 문서 일반화**
   - 한국어(AIDA, AIHUB, KRETA) + 영어(ChartQA, DocVQA) 동시 평가
   - 문서 레이아웃 기반 접근이 언어에 독립적임을 검증

---

# 7. 탑티어 가능성 근거

본 연구는:

* 단순 성능 향상이 아닌 **구조적 inductive bias** 제안 (novelty)
* Efficiency + Hallucination + Scaling **3요소 동시 분석** (comprehensiveness)
* **Encoder-level architectural novelty** — projector/LLM 수정 없이 vision encoder만으로 개선
* 문서 AI 분야에서 **명확한 gap** 해결 (uniform patching → adaptive patching)
* **5.5M rows** 규모의 내부 학습 데이터 + **12개 이상 공개 벤치마크** 평가
* 한-영 **다국어 실험**으로 일반화 능력 검증

따라서 CVPR/ICCV/NeurIPS submission 가능 수준의 기여를 목표로 한다.

---

# 8. 인프라 및 리소스 요약

| 항목 | 스펙 |
|------|------|
| GPU | A100 80GB × 4 (학습), A100 80GB × 1 (평가/분석) |
| 학습 데이터 총량 | ~400K 샘플 (5.5M에서 균형 샘플링) |
| Density Estimator 학습 | ~2시간 (A100 × 1) |
| Stage 2 학습 | ~24시간 (A100 × 4) |
| Stage 3 SCR 학습 | ~12시간 (A100 × 4) |
| 전체 벤치마크 평가 | ~6시간 (A100 × 1) |
| 데이터 경로 (juyeon) | `/NetDisk/juyeon/train/` — 차트/문서 원본 데이터 |
| 데이터 경로 (ingyu) | `/NetDisk/ingyu/VLM_DATA/mllm_ready/` — V3 라벨 + 이미지 |
| 코드 경로 | `/NetDisk/juyeon/DSA/` |
| 실험 로그 | `/NetDisk/juyeon/research/` (연구 워크플로우 연동) |

