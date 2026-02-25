"""
SCR (Structural Consistency Regularization) Loss Functions
===========================================================
Attention entropy regularization for text-dense patches.
Density estimator 출력으로 text-dense 영역을 식별하고,
해당 영역의 attention entropy를 최소화하여 구조적 집중도를 높입니다.

Usage:
    이 파일은 모듈로 import하여 16_scr_train.py에서 사용합니다.
"""

import torch
import torch.nn.functional as F


def create_density_mask(density_model, pixel_values, threshold=0.3,
                        grid_h=32, grid_w=32):
    """
    Run density estimator → binary mask of text-dense patches.

    Args:
        density_model: DensityEstimator (frozen, eval mode)
        pixel_values: [B, 3, 448, 448] tensor
        threshold: density value above which a patch is "text-dense"
        grid_h, grid_w: patch grid size (32×32 = 1024 patches)

    Returns:
        mask: [B, grid_h * grid_w] bool tensor, True = text-dense patch
    """
    with torch.no_grad():
        # Density estimator uses float32 (BatchNorm)
        density_maps = density_model(pixel_values.float())  # [B, 1, 28, 28]

    # Resize to patch grid
    density_grid = F.interpolate(
        density_maps, size=(grid_h, grid_w),
        mode="bilinear", align_corners=False,
    )  # [B, 1, grid_h, grid_w]

    # Flatten and threshold
    density_flat = density_grid.reshape(density_maps.shape[0], -1)  # [B, N]
    mask = density_flat > threshold  # [B, N]

    return mask


def compute_entropy_loss(attn_weights_list, density_mask):
    """
    Minimize attention entropy on text-dense patches.

    For text-dense query patches, we want the attention distribution to be
    sharper (lower entropy), encouraging focused attention on informative regions.

    Args:
        attn_weights_list: list of [B, H, N, N] tensors WITH gradient
            (from collect_scr_attn). N includes CLS token.
        density_mask: [B, N_spatial] bool tensor, True = text-dense patch
            N_spatial = 1024 (32×32 grid, no CLS token)

    Returns:
        entropy_loss: scalar tensor (with gradient)
    """
    if not attn_weights_list:
        return torch.tensor(0.0)

    eps = 1e-8
    total_entropy = 0.0
    count = 0

    for attn_w in attn_weights_list:
        # attn_w: [B, H, N, N] where N may include CLS token
        B, H, N, _ = attn_w.shape

        # Determine if CLS token is present
        N_spatial = density_mask.shape[1]  # 1024
        has_cls = (N == N_spatial + 1)

        if has_cls:
            # Skip CLS token (index 0), use spatial tokens only
            attn_spatial = attn_w[:, :, 1:, :]  # [B, H, N_spatial, N]
        else:
            attn_spatial = attn_w  # [B, H, N_spatial, N_spatial]

        # Compute entropy: H(α) = -Σ α_i log(α_i) per query patch
        entropy = -(attn_spatial * (attn_spatial + eps).log()).sum(dim=-1)
        # entropy: [B, H, N_spatial]

        # Apply density mask: only text-dense patches
        # density_mask: [B, N_spatial]
        mask_expanded = density_mask.unsqueeze(1).expand_as(entropy)  # [B, H, N_spatial]

        # Mean entropy over text-dense patches only
        masked_entropy = entropy * mask_expanded.float()
        n_dense = mask_expanded.float().sum().clamp(min=1.0)
        layer_entropy = masked_entropy.sum() / n_dense

        total_entropy = total_entropy + layer_entropy
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=attn_weights_list[0].device)

    return total_entropy / count
