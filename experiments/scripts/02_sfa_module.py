"""
Experiment 2: Structure-Factorized Attention (SFA) Module
=========================================================
InternVL3.5의 InternViT에 SFA를 삽입하는 모듈.
논문 Section 3.1의 핵심 구현.

S_ij = (Q_i K_j^T) / sqrt(d) + φ(s_i, s_j)

φ(s_i, s_j) = w_row · [row_i == row_j] + w_col · [col_i == col_j] + w_block · [block_i == block_j]

Usage:
    이 파일은 단독 실행이 아닌 모듈로 import하여 사용합니다.
    03_sfa_integration.py에서 InternViT의 attention에 이 모듈을 삽입합니다.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralBias(nn.Module):
    """
    Structure-Factorized Attention의 구조적 바이어스 φ(s_i, s_j).

    Patch grid (H_p × W_p)에서 row/column/block 관계를 계산하고
    learnable weight로 attention score에 더합니다.

    Args:
        num_patches_h: patch grid height (e.g., 28 for 448/16)
        num_patches_w: patch grid width
        num_heads: attention head 수
        num_blocks: block embedding 최대 수
    """
    def __init__(self, num_patches_h=28, num_patches_w=28, num_heads=16, num_blocks=16):
        super().__init__()
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.num_heads = num_heads

        # Per-head learnable weights for structural relations
        self.w_row = nn.Parameter(torch.zeros(num_heads))  # same-row bias
        self.w_col = nn.Parameter(torch.zeros(num_heads))  # same-column bias
        self.w_dist = nn.Parameter(torch.zeros(num_heads))  # distance decay

        # Block-level embedding (learned from density map)
        self.block_embed = nn.Embedding(num_blocks, num_heads)

        # Initialize small so it doesn't disrupt pretrained attention
        nn.init.normal_(self.w_row, std=0.02)
        nn.init.normal_(self.w_col, std=0.02)
        nn.init.normal_(self.w_dist, std=0.02)
        nn.init.normal_(self.block_embed.weight, std=0.02)

        # Precompute row/col indices
        self._precompute_indices()

    def _precompute_indices(self):
        """Patch 위치의 row, column 인덱스 미리 계산"""
        H, W = self.num_patches_h, self.num_patches_w
        N = H * W

        rows = torch.arange(H).unsqueeze(1).expand(H, W).reshape(N)  # [N]
        cols = torch.arange(W).unsqueeze(0).expand(H, W).reshape(N)  # [N]

        # Same-row indicator: [N, N]
        same_row = (rows.unsqueeze(1) == rows.unsqueeze(0)).float()
        # Same-col indicator: [N, N]
        same_col = (cols.unsqueeze(1) == cols.unsqueeze(0)).float()

        # Manhattan distance (normalized)
        row_dist = (rows.unsqueeze(1) - rows.unsqueeze(0)).float().abs()
        col_dist = (cols.unsqueeze(1) - cols.unsqueeze(0)).float().abs()
        manhattan = (row_dist + col_dist) / (H + W)  # normalized to ~[0, 1]

        self.register_buffer("same_row", same_row)
        self.register_buffer("same_col", same_col)
        self.register_buffer("manhattan_dist", manhattan)

    def forward(self, block_ids=None):
        """
        구조적 바이어스 행렬 계산.

        Args:
            block_ids: [N] tensor of block assignments (from density-based clustering)

        Returns:
            bias: [num_heads, N, N] structural bias to add to attention scores
        """
        N = self.num_patches_h * self.num_patches_w

        # Row bias: [num_heads, N, N]
        row_bias = self.w_row.view(-1, 1, 1) * self.same_row.unsqueeze(0)

        # Column bias: [num_heads, N, N]
        col_bias = self.w_col.view(-1, 1, 1) * self.same_col.unsqueeze(0)

        # Distance decay: [num_heads, N, N]
        dist_bias = -self.w_dist.view(-1, 1, 1).abs() * self.manhattan_dist.unsqueeze(0)

        bias = row_bias + col_bias + dist_bias

        # Block bias (if provided)
        if block_ids is not None:
            block_emb = self.block_embed(block_ids)  # [N, num_heads]
            # Same-block similarity
            block_sim = torch.einsum("ih,jh->hij", block_emb, block_emb)  # [num_heads, N, N]
            bias = bias + block_sim

        return bias


class SFAAttention(nn.Module):
    """
    Structure-Factorized Attention layer.
    기존 InternViT attention을 대체합니다.

    Standard: Attn = softmax(QK^T / sqrt(d)) V
    SFA:      Attn = softmax(QK^T / sqrt(d) + φ(s_i, s_j)) V
    """
    def __init__(self, dim, num_heads, num_patches_h=32, num_patches_w=32, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

        self.structural_bias = StructuralBias(
            num_patches_h=num_patches_h,
            num_patches_w=num_patches_w,
            num_heads=num_heads,
        )

    def forward(self, x, block_ids=None):
        """
        Args:
            x: [B, N, D] visual tokens (N may include CLS token)
            block_ids: [N] block assignments (optional)

        Returns:
            out: [B, N, D]
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: [B, H, N, d]

        # Content attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Add structural bias (only for spatial patch tokens, not CLS)
        N_spatial = self.structural_bias.num_patches_h * self.structural_bias.num_patches_w
        if N == N_spatial:
            struct_bias = self.structural_bias(block_ids)  # [H, N, N]
            attn = attn + struct_bias.unsqueeze(0)
        elif N == N_spatial + 1:
            # CLS token present: apply bias only to spatial tokens
            struct_bias = self.structural_bias(block_ids)  # [H, N_spatial, N_spatial]
            attn[:, :, 1:, 1:] = attn[:, :, 1:, 1:] + struct_bias.unsqueeze(0)
        # else: skip structural bias if size doesn't match (fallback to content-only)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Store attention for analysis (accessible via hook)
        self._last_attn_weights = attn.detach()

        return out


# ─── Attention Entropy Analysis ───

def compute_attention_entropy(attn_weights, patch_mask=None):
    """
    Attention entropy 계산 — 논문 Section 4 metric.

    Args:
        attn_weights: [B, H, N, N] attention weights
        patch_mask: [B, N] boolean mask for text-region patches (optional)

    Returns:
        entropy: scalar, mean entropy over all heads and text patches
    """
    # H(α) = -Σ α_i log(α_i)
    eps = 1e-8
    entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)  # [B, H, N]

    if patch_mask is not None:
        # Only compute entropy for text-region patches
        mask = patch_mask.unsqueeze(1).expand_as(entropy)  # [B, H, N]
        entropy = entropy[mask]

    return entropy.mean()


# ─── Test ───

if __name__ == "__main__":
    print("Testing SFA module...")

    B, N, D = 2, 784, 1024  # 28×28 patches, dim=1024
    num_heads = 16

    model = SFAAttention(dim=D, num_heads=num_heads)
    x = torch.randn(B, N, D)

    # Without block_ids
    out, attn = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Attn:   {attn.shape}")

    # With block_ids
    block_ids = torch.randint(0, 8, (N,))
    out2, attn2 = model(x, block_ids=block_ids)
    print(f"Output (with blocks): {out2.shape}")

    # Entropy
    ent = compute_attention_entropy(attn)
    print(f"Attention entropy: {ent.item():.4f}")

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    struct = sum(p.numel() for p in model.structural_bias.parameters())
    print(f"Total params: {total:,} | Structural bias params: {struct:,} ({struct/total*100:.1f}%)")
