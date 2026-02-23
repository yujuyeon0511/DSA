"""
SFA Architecture Diagram — 논문 Figure 2 (v2 refined)
======================================================
Usage:
    conda activate docmllm
    python experiments/scripts/gen_architecture_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# ── Colors ──
C = {
    "blue": "#1A73E8", "teal": "#00897B", "amber": "#F9AB00",
    "slate": "#70757A", "text": "#202124", "gray": "#5F6368",
    "ltblue": "#E8F0FE", "ltamber": "#FEF7E0", "ltteal": "#E0F2F1",
    "ltgray": "#F1F3F4", "green": "#4CAF50", "ltgreen": "#E8F5E9",
    "purple": "#7B1FA2", "ltpurple": "#F3E5F5",
    "orange": "#FF9800", "ltorange": "#FFF3E0",
    "yellow": "#FFF9C4",
}


def rbox(ax, x, y, w, h, label, fc="white", ec="#202124", lw=1.0,
         fs=7, bold=False, sub=None, sub_fs=5.5, tc=None):
    """Rounded box with centered label"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                         fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(box)
    col = tc or C["text"]
    weight = "bold" if bold else "normal"
    offset = 0.004 if sub else 0
    ax.text(x+w/2, y+h/2+offset, label, ha="center", va="center",
            fontsize=fs, color=col, fontweight=weight, zorder=5)
    if sub:
        ax.text(x+w/2, y+h/2-0.009, sub, ha="center", va="center",
                fontsize=sub_fs, color=C["gray"], zorder=5)


def arr(ax, x1, y1, x2, y2, c="#202124", lw=0.9, style="-|>"):
    """Arrow"""
    a = FancyArrowPatch((x1,y1),(x2,y2), arrowstyle=style, color=c,
                        lw=lw, mutation_scale=9, zorder=3)
    ax.add_patch(a)


def slbl(ax, x, y, txt, fs=5, c=None):
    """Shape label"""
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs,
            color=c or C["gray"], fontstyle="italic", zorder=6,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))


# ================================================================
fig = plt.figure(figsize=(22, 13))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

# ================================================================
# TITLE
# ================================================================
ax.text(0.50, 0.985, "Structure-Factorized Attention (SFA) for Document-Centric MLLMs",
        ha="center", va="top", fontsize=14, fontweight="bold", color=C["text"])
ax.text(0.50, 0.965, "Figure 2 — Architecture Overview",
        ha="center", va="top", fontsize=10, color=C["gray"])

# ================================================================
# LEGEND
# ================================================================
leg = [("■ Novel (SFA)", C["blue"]), ("■ Novel (ADAT)", C["amber"]),
       ("■ Trainable (4.0%)", "#1565C0"), ("□ Frozen (96.0%)", C["slate"])]
for i, (txt, col) in enumerate(leg):
    ax.text(0.03 + i*0.14, 0.945, txt, fontsize=7, color=col,
            fontweight="bold", va="center")

# ================================================================
# PIPELINE (top band, y ≈ 0.82)
# ================================================================
PY = 0.82  # pipeline y center
BH = 0.055

pipe = [
    # x,   w,     label,              sub,                   fc,          ec,          lw,  bold
    (0.015, 0.070, "Input\nImage",    "448×448×3",          "white",     C["text"],   1.0, False),
    (0.115, 0.100, "Density\nEstimator","CNN 186K params",  C["ltamber"],C["amber"],  2.0, True),
    (0.245, 0.110, "Adaptive Patch\nTokenization","ADAT",   C["ltamber"],C["amber"],  2.0, True),
    (0.390, 0.135, "InternViT-300M\n+ SFA","24 Layers",    C["ltblue"], C["blue"],   2.5, True),
    (0.560, 0.115, "Pixel Shuffle\n+ Projector","→256 tokens",C["ltblue"],C["blue"], 1.2, False),
    (0.710, 0.100, "Qwen3-8B\n(LLM)","36L · Frozen",      C["ltgray"], C["slate"],  1.2, False),
    (0.845, 0.055, "Output",          '"6.12%"',            "white",     C["text"],   1.0, False),
]

for (x,w,l,s,fc,ec,lw,bd) in pipe:
    rbox(ax, x, PY-BH/2, w, BH, l, fc=fc, ec=ec, lw=lw, fs=7.5, bold=bd, sub=s, sub_fs=5.5)

# Badges
badges = [
    (0.115,0.100,"★ NOVEL",C["amber"]), (0.245,0.110,"★ NOVEL",C["amber"]),
    (0.390,0.135,"★ NOVEL",C["blue"]),
    (0.115,0.100,"TRAIN",C["amber"]), (0.390,0.135,"TRAIN",C["blue"]),
    (0.560,0.115,"TRAIN",C["blue"]),
]
for (bx,bw,txt,col) in badges:
    if "NOVEL" in txt:
        ax.text(bx+0.004, PY+BH/2-0.005, "★", fontsize=7, color=col,
                ha="left", va="top", zorder=7)
    elif "TRAIN" in txt:
        ax.text(bx+bw/2, PY+BH/2+0.006, "TRAINABLE", ha="center", va="bottom",
                fontsize=4.5, color=col, fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=col, lw=0.5))

ax.text(0.710+0.050, PY+BH/2+0.006, "FROZEN", ha="center", va="bottom",
        fontsize=4.5, color=C["slate"], fontweight="bold", zorder=7,
        bbox=dict(boxstyle="round,pad=0.1", fc=C["ltgray"], ec=C["slate"], lw=0.5))

# Pipeline arrows
arrow_xs = [(0.085,0.115),(0.215,0.245),(0.355,0.390),(0.525,0.560),(0.675,0.710),(0.810,0.845)]
for (a,b) in arrow_xs:
    arr(ax, a, PY, b, PY, lw=1.3)

# Tensor shape labels
tlbls = [
    (0.100,"(B,3,448,448)"), (0.230,"D(x,y) 28×28"), (0.373,"(B,1025,1024)"),
    (0.543,"(B,1025,1024)"), (0.693,"(B,256,4096)"), (0.828,"text"),
]
for (tx,tl) in tlbls:
    slbl(ax, tx, PY+0.022, tl, fs=4.8)

# Density → SFA dashed arrow (block_ids)
ax.annotate("", xy=(0.430, PY-BH/2-0.002), xytext=(0.165, PY-BH/2-0.002),
            arrowprops=dict(arrowstyle="-|>", color=C["amber"], ls="--", lw=0.8,
                           connectionstyle="arc3,rad=-0.2"))
ax.text(0.30, PY-BH/2-0.025, "block_ids (clustering)", ha="center", fontsize=4.5,
        color=C["amber"], fontstyle="italic")

# ================================================================
# BOTTOM-LEFT: Encoder Layer Detail
# ================================================================
DX, DY, DW, DH = 0.015, 0.010, 0.48, 0.68
ax.add_patch(FancyBboxPatch((DX,DY), DW, DH, boxstyle="round,pad=0.008",
             fc="#FAFBFF", ec=C["blue"], lw=2.0, zorder=1))
ax.text(DX+DW/2, DY+DH-0.010, "InternViT Encoder Layer (×24)",
        ha="center", va="top", fontsize=10, fontweight="bold", color=C["blue"])

# Connector from pipeline
ax.annotate("", xy=(DX+DW/2, DY+DH), xytext=(0.457, PY-BH/2),
            arrowprops=dict(arrowstyle="-", color=C["blue"], ls=":", lw=1.0))

# Internal flow
cx = DX + 0.16  # center x
bw = 0.20
bh_s = 0.025  # small box height
sp = 0.040    # spacing

yy = DY + DH - 0.050

# Input
rbox(ax, cx-bw/2, yy, bw, bh_s, "Input: (B, 1025, 1024)", fs=6.5)
slbl(ax, cx+bw/2+0.030, yy+bh_s/2, "1 CLS + 32×32 spatial", fs=4.5)

# Save residual start
res_start_y = yy + bh_s/2

yy -= sp*0.65
arr(ax, cx, yy+sp*0.65, cx, yy+bh_s)

# LayerNorm1
rbox(ax, cx-bw/2, yy, bw, bh_s, "LayerNorm (1024)", fc=C["ltpurple"], ec=C["purple"], fs=6.5)

yy -= sp*0.55
arr(ax, cx, yy+sp*0.55, cx, yy+bh_s)

# ── SFA Attention Block ──
sfa_top = yy + bh_s + 0.005
sfa_h = 0.30
sfa_w = bw + 0.10
sfa_x = cx - sfa_w/2
sfa_y = sfa_top - sfa_h

ax.add_patch(FancyBboxPatch((sfa_x, sfa_y), sfa_w, sfa_h,
             boxstyle="round,pad=0.005", fc="#E8F0FE", ec=C["blue"], lw=2.0, zorder=1.5))
ax.text(sfa_x+sfa_w/2, sfa_top-0.008, "Structure-Factorized Attention (SFA)",
        ha="center", va="top", fontsize=8, fontweight="bold", color=C["blue"], zorder=6)

# QKV
qy = sfa_top - 0.045
qw = 0.17
rbox(ax, cx-qw/2, qy, qw, 0.022, "QKV Linear (1024→3072)", fc="white", ec=C["text"], fs=6, lw=0.7)

# Q K V split
sy = qy - 0.035
offsets = [-0.055, 0, 0.055]
for letter, off in zip(["Q","K","V"], offsets):
    rbox(ax, cx+off-0.020, sy, 0.040, 0.020, letter, fc="white", ec=C["blue"],
         fs=7, bold=True, lw=0.8)
    arr(ax, cx + off*0.6, qy, cx+off, sy+0.020, c=C["text"], lw=0.5)

slbl(ax, cx+0.11, sy+0.010, "each (B,16,N,64)", fs=4.5)

# Content Attn + Structural Bias (side by side)
aby = sy - 0.050
ab_w = 0.095
content_cx = cx - 0.055
struct_cx = cx + 0.055

rbox(ax, content_cx-ab_w/2, aby, ab_w, 0.035, "Content\nAttention", fc="white",
     ec=C["text"], fs=6, sub="QKᵀ / √d", sub_fs=5.5, lw=0.8)
rbox(ax, struct_cx-ab_w/2, aby, ab_w, 0.035, "Structural\nBias φ", fc=C["ltteal"],
     ec=C["teal"], fs=6, bold=True, sub="(16, 1024, 1024)", sub_fs=5, lw=1.5)

# Arrows Q,K → content
arr(ax, cx-0.055, sy, content_cx, aby+0.035, c=C["text"], lw=0.5)
arr(ax, cx, sy, content_cx+0.02, aby+0.035, c=C["text"], lw=0.5)

# ADD circle
add_y = aby - 0.027
ax.text(cx, add_y+0.012, "⊕", ha="center", va="center", fontsize=13,
        color=C["blue"], fontweight="bold", zorder=6,
        bbox=dict(boxstyle="circle,pad=0.06", fc="white", ec=C["blue"], lw=1.2))

arr(ax, content_cx, aby, cx-0.008, add_y+0.022, c=C["text"], lw=0.6)
arr(ax, struct_cx, aby, cx+0.008, add_y+0.022, c=C["teal"], lw=0.6)

# Equation box
eq_y = add_y - 0.020
ax.text(cx, eq_y,
        r"$S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}} + \varphi(s_i, s_j)$",
        ha="center", va="center", fontsize=8, color=C["text"], zorder=6,
        bbox=dict(boxstyle="round,pad=0.18", fc=C["yellow"], ec=C["amber"], lw=1.0))

# Softmax
sf_y = eq_y - 0.030
rbox(ax, cx-0.055, sf_y, 0.11, 0.020, "Softmax → Attn × V", fc="white",
     ec=C["text"], fs=6, lw=0.7)
arr(ax, cx, eq_y-0.012, cx, sf_y+0.020, c=C["text"], lw=0.6)

# Proj
pj_y = sf_y - 0.028
rbox(ax, cx-0.055, pj_y, 0.11, 0.020, "Proj (1024→1024)", fc="white",
     ec=C["text"], fs=6, lw=0.7)
arr(ax, cx, sf_y, cx, pj_y+0.020, c=C["text"], lw=0.6)

# ── End SFA block ──

# Residual 1 (+)
r1_y = sfa_y - 0.028
ax.text(cx, r1_y+0.012, "+", ha="center", va="center", fontsize=10,
        color=C["green"], fontweight="bold", zorder=6,
        bbox=dict(boxstyle="circle,pad=0.05", fc=C["ltgreen"], ec=C["green"], lw=1.0))
arr(ax, cx, sfa_y, cx, r1_y+0.022, c=C["text"], lw=0.7)
ax.text(cx+0.07, r1_y+0.012, "× LayerScale₁", fontsize=5, color=C["gray"], va="center")

# Residual skip arrow (left side)
skip_x = sfa_x - 0.025
ax.plot([cx-bw/2-0.003, skip_x, skip_x, cx-0.018],
        [res_start_y, res_start_y, r1_y+0.012, r1_y+0.012],
        color=C["green"], lw=1.0, ls="--", zorder=3)
arr(ax, skip_x, r1_y+0.012, cx-0.018, r1_y+0.012, c=C["green"], lw=0.8)
ax.text(skip_x-0.006, (res_start_y+r1_y+0.012)/2, "Residual", fontsize=4.5,
        color=C["green"], rotation=90, ha="center", va="center")

# LayerNorm2
ln2_y = r1_y - 0.028
rbox(ax, cx-bw/2, ln2_y, bw, bh_s, "LayerNorm (1024)", fc=C["ltpurple"],
     ec=C["purple"], fs=6.5, lw=0.7)
arr(ax, cx, r1_y, cx, ln2_y+bh_s, c=C["text"], lw=0.7)

# MLP
mlp_y = ln2_y - 0.042
rbox(ax, cx-bw/2, mlp_y, bw, 0.032, "MLP (FFN)", fc=C["ltorange"],
     ec=C["orange"], fs=7, bold=True,
     sub="Linear(1024→4096) → GELU → Linear(4096→1024)", sub_fs=5, lw=1.0)
arr(ax, cx, ln2_y, cx, mlp_y+0.032, c=C["text"], lw=0.7)

# Residual 2
r2_y = mlp_y - 0.028
ax.text(cx, r2_y+0.012, "+", ha="center", va="center", fontsize=10,
        color=C["green"], fontweight="bold", zorder=6,
        bbox=dict(boxstyle="circle,pad=0.05", fc=C["ltgreen"], ec=C["green"], lw=1.0))
arr(ax, cx, mlp_y, cx, r2_y+0.022, c=C["text"], lw=0.7)
ax.text(cx+0.07, r2_y+0.012, "× LayerScale₂", fontsize=5, color=C["gray"], va="center")

# Skip 2
skip2_x = sfa_x - 0.015
ax.plot([cx-bw/2+0.002, skip2_x, skip2_x, cx-0.018],
        [r1_y+0.012-0.005, r1_y+0.012-0.005, r2_y+0.012, r2_y+0.012],
        color=C["green"], lw=0.8, ls="--", zorder=3)

# Output
out_y = r2_y - 0.028
rbox(ax, cx-bw/2, out_y, bw, bh_s, "Output: (B, 1025, 1024)", fs=6.5)
arr(ax, cx, r2_y, cx, out_y+bh_s, c=C["text"], lw=0.7)

# ================================================================
# BOTTOM-RIGHT: Structural Bias Detail
# ================================================================
PX, PY2, PW, PH = 0.52, 0.010, 0.47, 0.68
ax.add_patch(FancyBboxPatch((PX,PY2), PW, PH, boxstyle="round,pad=0.008",
             fc="#E8F5F3", ec=C["teal"], lw=2.0, zorder=1))
ax.text(PX+PW/2, PY2+PH-0.010,
        "Structural Bias  φ(sᵢ, sⱼ)  —  Detailed View",
        ha="center", va="top", fontsize=10, fontweight="bold", color=C["teal"])

# Dashed connector from SFA block
ax.annotate("", xy=(PX, PY2+PH*0.65),
            xytext=(DX+DW, sfa_y+sfa_h*0.45),
            arrowprops=dict(arrowstyle="-|>", color=C["teal"], ls="--", lw=1.5,
                           connectionstyle="arc3,rad=0.08"))

# Full formula
fy = PY2 + PH - 0.050
ax.text(PX+PW/2, fy,
        r"$\varphi(s_i, s_j) = w_{row} \cdot [r_i\!=\!r_j]"
        r" + w_{col} \cdot [c_i\!=\!c_j]"
        r" + w_{dist} \cdot (-d_{ij})"
        r" + e_{b_i}^T e_{b_j}$",
        ha="center", va="center", fontsize=8.5, color=C["text"], zorder=6,
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=C["teal"], lw=0.8))
# Sub-labels for each term
term_labels = ["Row", "Column", "Distance", "Block"]
term_xs = [PX+0.10, PX+0.205, PX+0.315, PX+0.415]
for tx, tl in zip(term_xs, term_labels):
    ax.text(tx, fy-0.015, tl, ha="center", va="top", fontsize=5,
            color=C["teal"], fontweight="bold", zorder=6)

# ── 4 Component cards ──
card_y = fy - 0.095
card_w = 0.100
card_h = 0.070
card_gap = 0.008
card_x0 = PX + 0.020

comps = [
    ("Same-Row\nBias", "w_row · 𝟙[rᵢ = rⱼ]", "w_row: (16,) per-head\ninit: N(0, 0.02)", C["teal"]),
    ("Same-Col\nBias", "w_col · 𝟙[cᵢ = cⱼ]", "w_col: (16,) per-head\ninit: N(0, 0.02)", C["blue"]),
    ("Distance\nDecay", "w_dist · (−dᵢⱼ)", "w_dist: (16,) per-head\nmanhattan, norm [0,1]", C["amber"]),
    ("Block\nSimilarity", "eᵢᵀ · eⱼ", "Embedding(16, 16)\n256 params", C["purple"]),
]

for i, (title, formula, desc, col) in enumerate(comps):
    x = card_x0 + i*(card_w+card_gap)
    ax.add_patch(FancyBboxPatch((x, card_y), card_w, card_h,
                 boxstyle="round,pad=0.004", fc="white", ec=col, lw=1.5, zorder=2))
    ax.text(x+card_w/2, card_y+card_h-0.008, title, ha="center", va="top",
            fontsize=6, fontweight="bold", color=col, zorder=6)
    ax.text(x+card_w/2, card_y+card_h/2+0.002, formula, ha="center", va="center",
            fontsize=6, color=C["text"], zorder=6)
    ax.text(x+card_w/2, card_y+0.010, desc, ha="center", va="bottom",
            fontsize=4.5, color=C["gray"], zorder=6, linespacing=1.3)

# ── Heatmap visualizations ──
hm_y = card_y - 0.140
hm_s = 0.100  # heatmap square size

np.random.seed(42)
# Row bias: horizontal bands
row = np.zeros((32,32))
for r in range(32):
    row[r,:] = 1.0 if r % 3 < 2 else 0.15
# Col bias: vertical bands
col = np.zeros((32,32))
for c in range(32):
    col[:,c] = 1.0 if c % 3 < 2 else 0.15
# Distance: radial
yg, xg = np.mgrid[0:32,0:32]
dist = 1.0 - np.sqrt((xg-16)**2+(yg-16)**2)/22.0
dist = np.clip(dist, 0, 1)
# Combined
comb = 0.35*row + 0.35*col + 0.30*dist
comb /= comb.max()

hmaps = [
    (row, "Row Bias", "YlOrRd"),
    (col, "Column Bias", "Blues"),
    (dist, "Distance Decay", "inferno"),
    (comb, "Combined φ", "RdBu_r"),
]

for i, (data, title, cmap) in enumerate(hmaps):
    hx = card_x0 + i*(card_w+card_gap) + (card_w-hm_s)/2
    ext = [hx, hx+hm_s, hm_y, hm_y+hm_s]
    ax.imshow(data, extent=ext, cmap=cmap, aspect="auto",
              interpolation="bilinear", zorder=4, alpha=0.9)
    ax.plot([hx, hx+hm_s, hx+hm_s, hx, hx],
            [hm_y, hm_y, hm_y+hm_s, hm_y+hm_s, hm_y],
            color=C["text"], lw=0.6, zorder=5)
    ax.text(hx+hm_s/2, hm_y-0.006, title, ha="center", va="top",
            fontsize=6, color=C["text"], fontweight="bold", zorder=6)
    # Arrow from card to heatmap
    arr(ax, card_x0+i*(card_w+card_gap)+card_w/2, card_y,
        hx+hm_s/2, hm_y+hm_s, c=C["gray"], lw=0.4)

# ── CLS note ──
ax.text(PX+PW/2, hm_y-0.025,
        "※ φ applied to spatial tokens only (index 1:1024).  CLS token (index 0): no structural bias.",
        ha="center", va="top", fontsize=5.5, color=C["teal"], fontstyle="italic")

# ── Parameter summary box ──
tb_y = hm_y - 0.075
tb_w = PW - 0.06
tb_h = 0.045
ax.add_patch(FancyBboxPatch((PX+0.03, tb_y), tb_w, tb_h,
             boxstyle="round,pad=0.004", fc="white", ec=C["text"], lw=0.8, zorder=2))
ax.text(PX+0.03+tb_w/2, tb_y+tb_h-0.005, "Parameter Summary",
        ha="center", va="top", fontsize=7, fontweight="bold", color=C["text"], zorder=6)

rows_t = [
    ("SFA Bias / Layer:","304  (w_row:16 + w_col:16 + w_dist:16 + block_embed:256)"),
    ("SFA Total (×24 layers):","7,296 params  (0.002% of 8.5B)"),
    ("Trainable:","337.6M  (4.0%)  =  Vision Encoder (300M) + Projector + SFA bias"),
]
for j, (k,v) in enumerate(rows_t):
    ty = tb_y + tb_h - 0.015 - j*0.011
    ax.text(PX+0.045, ty, k, ha="left", fontsize=5.5, color=C["text"],
            fontweight="bold", zorder=6)
    ax.text(PX+0.20, ty, v, ha="left", fontsize=5.5, color=C["gray"], zorder=6)

# ================================================================
# Pixel Shuffle detail strip (between pipeline and detail panels)
# ================================================================
ps_y = PY2 + PH + 0.010
ps_h = 0.045

ax.add_patch(FancyBboxPatch((PX, ps_y), PW, ps_h,
             boxstyle="round,pad=0.004", fc=C["ltblue"], ec=C["blue"], lw=1.0, zorder=2))
ax.text(PX+0.010, ps_y+ps_h-0.006, "Pixel Shuffle + MLP Projector (mlp1)", ha="left",
        fontsize=7, fontweight="bold", color=C["blue"], zorder=6)
ps_txt = (
    "(B,1025,1024) → drop CLS → (B,1024,1024) → reshape (B,32,32,1024)\n"
    "→ pixel_shuffle(0.5) → (B,16,16,4096) → flatten → (B,256,4096)\n"
    "→ LN(4096) → Linear(4096→4096) → GELU → Linear(4096→4096) → (B,256,4096)"
)
ax.text(PX+0.010, ps_y+0.005, ps_txt, ha="left", va="bottom",
        fontsize=5, color=C["text"], family="monospace", zorder=6, linespacing=1.4)

# ================================================================
# SAVE
# ================================================================
out_dir = "/NetDisk/juyeon/DSA/experiments/figures"
os.makedirs(out_dir, exist_ok=True)

fig.savefig(os.path.join(out_dir, "fig2_architecture.pdf"), bbox_inches="tight", pad_inches=0.08, dpi=300)
fig.savefig(os.path.join(out_dir, "fig2_architecture.png"), bbox_inches="tight", pad_inches=0.08, dpi=300)
plt.close(fig)
print(f"Saved: {out_dir}/fig2_architecture.{{pdf,png}}")
