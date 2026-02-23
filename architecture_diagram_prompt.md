# SFA Architecture Diagram AI Generation Prompt

> **용도**: AI 도구 (Claude, GPT-4o, Gemini 등)에 입력하여 논문 Figure 2 (Architecture Diagram) 초안 생성
> **출력 형식**: SVG / TikZ / draw.io XML / Mermaid 중 택1
> **최종 목표**: ECCV/CVPR 수준의 학술 논문 아키텍처 그림

---

## Prompt (English — for AI diagram generation)

아래 프롬프트를 AI에 그대로 입력하세요. 필요에 따라 출력 형식을 지정하면 됩니다.

---

### Main Prompt

```
Create a detailed, publication-quality architecture diagram for a research paper titled
"Structure-Factorized Attention for Document-Centric Multimodal LLMs".

The diagram should show the COMPLETE forward pass pipeline from input document image to
text output, with emphasis on the two novel modules: (1) Adaptive Density-Aware Tokenization
(ADAT) and (2) Structure-Factorized Attention (SFA).

============================================================
OVERALL PIPELINE (left-to-right flow):
============================================================

The diagram flows LEFT → RIGHT through these major stages:

[Input Image] → [Density Estimator] → [Adaptive Patch Tokenization] → [Vision Encoder w/ SFA] → [Pixel Shuffle + MLP Projector] → [LLM] → [Text Output]

Use a clean, horizontal layout. Each major block should be a rounded rectangle.
Draw data flow arrows between blocks with tensor shape annotations on each arrow.

============================================================
STAGE 1: INPUT
============================================================

- Show a small document/chart image thumbnail (448×448 px)
- Label: "Document Image"
- Tensor annotation on output arrow: "(B, 3, 448, 448)"

============================================================
STAGE 2: DENSITY ESTIMATOR (Novel Module — highlight with AMBER border #F9AB00)
============================================================

This is a lightweight CNN that predicts text density.

Internal structure (show as a vertical stack inside a dashed box):
  ┌─────────────────────────────────────────┐
  │  Text Density Estimator (186K params)   │
  │                                         │
  │  Conv2d(3→32, k=3, s=2) + BN + ReLU    │  448→224
  │  Conv2d(32→64, k=3, s=2) + BN + ReLU   │  224→112
  │  Conv2d(64→128, k=3, s=2) + BN + ReLU  │  112→56
  │  Conv2d(128→64, k=3, s=2) + BN + ReLU  │  56→28
  │  Conv2d(64→32, k=3, s=1) + BN + ReLU   │  28→28
  │  Conv2d(32→1, k=1) + Sigmoid           │  28→28
  │                                         │
  │  Output: D(x,y) ∈ [0,1]  (28×28)       │
  └─────────────────────────────────────────┘

- Input arrow: "(B, 3, 448, 448)" from the input image
- Output: Show a small density heatmap thumbnail (28×28, use yellow-red gradient)
- The density map feeds into BOTH:
  (a) Adaptive Patch Tokenizer (determines patch sizes)
  (b) Block ID assignment for SFA (optional, via clustering)

============================================================
STAGE 3: ADAPTIVE PATCH TOKENIZATION (Novel Module — highlight with AMBER border)
============================================================

Show this as a branching process:

  Density Map D(x,y) → Patch Size Assignment:
    - D > 0.7 (high density): 8×8 patches  (small, more tokens)
    - 0.3 < D ≤ 0.7 (medium):  14×14 patches (standard)
    - D ≤ 0.3 (low density):  32×32 patches (large, fewer tokens)

  → Token Budget Constraint: Σ (s_k² × Area_k) ≤ N

Show a visual: the same document image with a multi-scale grid overlay where
dense text regions have finer grids and blank areas have coarser grids.

Output arrow annotation: "(B, N, D_vit)" where N ≤ 1024, D_vit = 1024

NOTE: For the baseline (current implementation), uniform 14×14 patches are used,
producing 32×32 = 1024 patches. The adaptive version is the proposed contribution.

============================================================
STAGE 4: VISION ENCODER WITH SFA (Core — highlight with BLUE border #1A73E8)
============================================================

This is InternViT-300M modified with SFA. Show it as a tall vertical stack of layers.

Header label: "InternViT-300M + SFA (24 Layers)"

Show the internal structure of ONE encoder layer in an expanded/zoomed view:

  ┌──────────────────────────────────────────────────────┐
  │         InternVisionEncoderLayer (×24)                │
  │                                                       │
  │   Input: (B, 1025, 1024)  [1 CLS + 32×32 patches]   │
  │      │                                                │
  │      ▼                                                │
  │   ┌─────────┐                                         │
  │   │LayerNorm│  (1024)                                 │
  │   └────┬────┘                                         │
  │        │                                              │
  │        ▼                                              │
  │   ╔════════════════════════════════════════════╗       │
  │   ║  Structure-Factorized Attention (SFA)      ║      │
  │   ║                                            ║      │
  │   ║  ┌──────────┐                              ║      │
  │   ║  │ QKV Linear│ (1024 → 3072)              ║      │
  │   ║  └──┬───┬───┬┘                             ║      │
  │   ║     Q   K   V   each: (B, 16, 1025, 64)   ║      │
  │   ║     │   │                                  ║      │
  │   ║     ▼   ▼                                  ║      │
  │   ║  ┌────────────────┐  ┌──────────────────┐  ║      │
  │   ║  │Content Attention│  │Structural Bias φ │  ║      │
  │   ║  │ QK^T / √d      │  │                  │  ║      │
  │   ║  │(B,16,1025,1025)│  │(16, 1024, 1024)  │  ║      │
  │   ║  └───────┬────────┘  └────────┬─────────┘  ║      │
  │   ║          │         ╋ ADD       │            ║      │
  │   ║          └─────────┬───────────┘            ║      │
  │   ║                    ▼                        ║      │
  │   ║  S_ij = QK^T/√d + φ(s_i, s_j)             ║      │
  │   ║  [spatial tokens only, CLS excluded]        ║      │
  │   ║                    ▼                        ║      │
  │   ║              ┌──────────┐                   ║      │
  │   ║              │ Softmax  │                   ║      │
  │   ║              └────┬─────┘                   ║      │
  │   ║                   ▼                         ║      │
  │   ║          Attn × V → Proj (1024→1024)        ║      │
  │   ║                                            ║      │
  │   ╚════════════════════════════════════════════╝       │
  │        │                                              │
  │        × LayerScale (ls1)                              │
  │        │                                              │
  │   ─────┤ (+) Residual ◄───── Input                    │
  │        │                                              │
  │        ▼                                              │
  │   ┌─────────┐                                         │
  │   │LayerNorm│                                         │
  │   └────┬────┘                                         │
  │        ▼                                              │
  │   ┌──────────────────────────┐                        │
  │   │ MLP (FFN)                │                        │
  │   │ Linear(1024→4096) → GELU │                        │
  │   │ Linear(4096→1024)        │                        │
  │   └────────────┬─────────────┘                        │
  │        × LayerScale (ls2)                              │
  │        │                                              │
  │   ─────┤ (+) Residual ◄───── LayerNorm output         │
  │        │                                              │
  │   Output: (B, 1025, 1024)                              │
  └──────────────────────────────────────────────────────┘

============================================================
STAGE 4-DETAIL: STRUCTURAL BIAS φ(s_i, s_j) — ZOOMED INSET
============================================================

Show a separate detailed inset box (connected by a dashed line to the φ block above)
explaining the structural bias computation:

  ╔══════════════════════════════════════════════════════════╗
  ║  Structural Bias φ(s_i, s_j)  [304 params per layer]   ║
  ║                                                         ║
  ║  φ = w_row · 𝟙[row_i = row_j]       ← Same-Row Bias   ║
  ║    + w_col · 𝟙[col_i = col_j]       ← Same-Column Bias║
  ║    + w_dist · (-manhattan(i,j))      ← Distance Decay  ║
  ║    + block_embed(b_i)ᵀ block_embed(b_j) ← Block Sim   ║
  ║                                                         ║
  ║  Parameters:                                            ║
  ║    w_row:  (16,)     ← per-head, init N(0, 0.02)       ║
  ║    w_col:  (16,)     ← per-head, init N(0, 0.02)       ║
  ║    w_dist: (16,)     ← per-head, init N(0, 0.02)       ║
  ║    block_embed: Embedding(16 blocks, 16 heads)          ║
  ║                                                         ║
  ║  Precomputed buffers (32×32 grid):                      ║
  ║    same_row:      (1024, 1024)  binary indicator        ║
  ║    same_col:      (1024, 1024)  binary indicator        ║
  ║    manhattan_dist: (1024, 1024) normalized [0,1]        ║
  ║                                                         ║
  ║  Applied to: attn[:, :, 1:, 1:]  (spatial tokens only)  ║
  ║  CLS token: no structural bias applied                  ║
  ╚══════════════════════════════════════════════════════════╝

Alongside this box, show 4 small matrix heatmaps (conceptual):
  [Row Bias]  [Col Bias]  [Distance]  [Combined φ]
Each should be a 32×32 thumbnail with intuitive patterns:
  - Row Bias: horizontal bands (same-row patches highlighted)
  - Col Bias: vertical bands (same-column patches highlighted)
  - Distance: radial gradient from center (closer = stronger)
  - Combined: grid-like pattern with both row and column structure

============================================================
STAGE 5: PIXEL SHUFFLE DOWNSAMPLING + MLP PROJECTOR
============================================================

  ┌────────────────────────────────────────────────────┐
  │  Feature Extraction & Projection                   │
  │                                                    │
  │  1. Remove CLS token                               │
  │     (B, 1025, 1024) → (B, 1024, 1024)             │
  │                                                    │
  │  2. Reshape to spatial grid                        │
  │     (B, 1024, 1024) → (B, 32, 32, 1024)           │
  │                                                    │
  │  3. Pixel Shuffle (downsample_ratio=0.5)           │
  │     (B, 32, 32, 1024) → (B, 16, 16, 4096)         │
  │     [4× spatial reduction, 4× channel expansion]   │
  │                                                    │
  │  4. Flatten to sequence                            │
  │     (B, 16, 16, 4096) → (B, 256, 4096)            │
  │                                                    │
  │  5. MLP Projector (mlp1):                          │
  │     LayerNorm(4096)                                │
  │     → Linear(4096, 4096) → GELU                   │
  │     → Linear(4096, 4096)                           │
  │     Output: (B, 256, 4096)                         │
  └────────────────────────────────────────────────────┘

Output arrow: "(B, 256, 4096) — 256 visual tokens"

============================================================
STAGE 6: LANGUAGE MODEL (LLM)
============================================================

  ┌─────────────────────────────────────────┐
  │  Qwen3-8B (Frozen)                      │
  │                                         │
  │  36 Transformer Layers                  │
  │  hidden_size = 4096                     │
  │  32 attention heads (GQA, 8 KV heads)   │
  │  head_dim = 128                         │
  │  FFN intermediate = 12288               │
  │  Activation: SiLU                       │
  │  Vocab: 151,936                         │
  │                                         │
  │  Input: [System] [Visual Tokens] [Query]│
  │         ↑                               │
  │    256 visual tokens embedded at 4096   │
  │    replace <IMG_CONTEXT> placeholders   │
  │                                         │
  │  FROZEN — no gradient flows back        │
  └─────────────────────────────────────────┘

  → Output: "6.12%"  (text answer to the question)

============================================================
TRAINABLE vs FROZEN INDICATORS
============================================================

Use visual cues to distinguish trainable and frozen components:
  - TRAINABLE: Solid border, filled background (light blue tint)
    → Density Estimator (186K params)
    → Vision Encoder attention layers (SFA: 7,296 new params + 300M pretrained)
    → MLP Projector mlp1
  - FROZEN: Dashed border, gray background
    → LLM (Qwen3-8B, ~8B params)

Show a small legend:
  ■ Trainable (4.0% of total)  □ Frozen (96.0%)
  ★ Novel module (SFA / ADAT)

============================================================
COLOR SCHEME
============================================================

Use these exact colors for consistency with the paper:
  - SFA / Ours blocks:    #1A73E8 (Research Blue) — borders and highlights
  - Structural bias φ:    #00897B (Structural Teal) — φ inset box
  - Density / ADAT:       #F9AB00 (Density Amber) — density estimator border
  - Baseline / Standard:  #70757A (Neutral Slate) — frozen LLM, standard layers
  - Background:           #F8F9FA (Paper White) — clean white background
  - Text / Labels:        #202124 (Charcoal) — all text and arrows
  - Tensor annotations:   #5F6368 (gray) — shape labels on arrows

============================================================
STYLE GUIDELINES
============================================================

  - Academic publication quality (ECCV/CVPR style)
  - Clean, minimal design with NO unnecessary decoration
  - Font: sans-serif (Arial or Helvetica)
  - Font size: Module names 11pt, tensor shapes 8pt, math 10pt
  - Arrow style: thin solid lines with arrowheads
  - Module boxes: rounded corners (4px radius), thin borders (1pt)
  - Novel modules: slightly thicker border (2pt) in their designated color
  - Mathematical notation rendered properly: subscripts, superscripts, √d
  - Aspect ratio: approximately 3:1 (wide landscape for double-column paper)
  - Total width should fit a double-column figure (~6.875 inches)

============================================================
EQUATION TO INCLUDE IN THE DIAGRAM
============================================================

Place this equation prominently near the SFA block:

  S_ij = (Q_i · K_j^T) / √d  +  φ(s_i, s_j)
         \_____________/         \____________/
          Content Attn         Structural Bias

============================================================
PARAMETER SUMMARY (include as a small table or annotation)
============================================================

| Component            | New Params  | Note              |
|----------------------|-------------|-------------------|
| Density Estimator    | 186K        | Lightweight CNN   |
| SFA Bias (per layer) | 304         | w_row+w_col+w_dist+block_embed |
| SFA Total (24 layers)| 7,296       | 0.002% of model   |
| Trainable Total      | 337.6M      | 4.0% of 8.5B      |
```

---

## TikZ 버전 요청 시 추가 프롬프트

```
Generate TikZ/LaTeX code for this architecture diagram.
Use the following TikZ libraries: positioning, arrows.meta, fit, calc, backgrounds, shapes.geometric
Define custom colors matching the hex codes above.
Use \footnotesize for tensor annotations and \small for module names.
The diagram should compile with pdflatex without errors.
```

---

## Mermaid 버전 요청 시 추가 프롬프트

```
Generate a Mermaid diagram (flowchart LR) for this architecture.
Use subgraphs for each major stage.
Add notes for tensor shapes.
Style novel modules with the specified colors.
```

---

## draw.io XML 요청 시 추가 프롬프트

```
Generate draw.io compatible XML for this architecture diagram.
Use proper styling with the color scheme above.
Group related components.
Make it editable for fine-tuning the layout.
```

---

## 주의사항

1. **실제 구현과 일치해야 하는 수치**:
   - InternViT: 24 layers, dim=1024, 16 heads, head_dim=64, patch=14, image=448
   - Grid: 32×32 = 1024 patches + 1 CLS = 1025 tokens
   - Pixel Shuffle: 32×32×1024 → 16×16×4096 → flatten → 256 tokens
   - MLP Projector: LayerNorm(4096) → Linear(4096→4096) → GELU → Linear(4096→4096)
   - LLM: Qwen3-8B, 36 layers, dim=4096, 32 heads (GQA 8 KV), FFN=12288
   - SFA bias: 304 params/layer (w_row:16 + w_col:16 + w_dist:16 + block_embed:16×16=256)

2. **SFA 적용 위치**: InternViT 내부의 각 layer attention (24개 전부)
   - CLS 토큰(index 0)에는 structural bias 미적용
   - spatial tokens (index 1~1024)에만 적용: `attn[:, :, 1:, 1:] += φ`

3. **Density Estimator**:
   - 별도 CNN (InternViT 외부), self-supervised 학습
   - 출력 density map은 ADAT에 사용 (patch size 결정)
   - 또한 block_id 생성에도 활용 가능 (clustering → SFA의 block bias)

4. **학습 전략**:
   - Stage 1: Density Estimator 독립 학습 (MSE loss, pseudo label)
   - Stage 2: Vision Encoder (SFA injected) + Projector fine-tuning (LLM frozen)
   - Stage 3: SCR loss 추가 fine-tuning (L_entropy + L_grounding + L_stability)

5. **논문 Figure 2와의 관계**:
   - 이 다이어그램이 Figure 2의 완성본
   - 왼쪽: 입력 → 중간: 핵심 모듈 (ADAT, SFA) → 오른쪽: 출력
   - SFA 내부 수식 zoomed inset이 핵심 기여를 시각적으로 강조
