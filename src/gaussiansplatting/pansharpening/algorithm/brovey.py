import torch
from .pansharp_utils import torch_interpolation, resize_tensor


def simple_brovey(img_pan: torch.Tensor, img_msi: torch.Tensor) -> torch.Tensor:
    """Apply the Brovey fusion algorithm with resizing.

    Parameters:
    - img_pan: tensor of shape (H, W, 1)
    - img_msi: tensor of shape (h, w, C)

    Returns:
    - fused: tensor of shape (H, W, C)
    """
    # # Permute HWC to CHW
    # img_msi_chw = img_msi.permute(2, 0, 1).unsqueeze(0)  # [1, C, h, w]
    # img_pan_chw = img_pan.permute(2, 0, 1).squeeze(0)    # [H, W]

    # Upsample MSI to PAN size
    img_msi_up = torch_interpolation(img_msi_dw=img_msi, img_pan_up=img_pan).squeeze(0)

    # Brovey formula
    eps = 1e-8
    sum_msi = img_msi_up.sum(dim=0, keepdim=True)  # [1, H, W]
    ratio = img_pan.unsqueeze(0) / (sum_msi + eps)  # [1, H, W]
    fused = img_msi_up * ratio  # [C, H, W]

    return fused


def brovey_pansharp(img_pan, img_msi, W=0.1):
    """
    Brovey pansharpening algorithm
    img_pan: torch tensor of shape (H, W)
    img_msi: torch tensor of shape (H_ms, W_ms, C)
    """
    if img_pan.ndim == 3:
        img_pan = img_pan.squeeze()  # Assume (H, W, 1) -> (H, W)

    img_msi_t = img_msi  # .permute(2, 0, 1)  # (C, H, W)
    rescaled_ms = resize_tensor(img_msi_t, size_out=img_pan.shape)  # (C, H, W)
    denominator = (W * rescaled_ms.sum(dim=0, keepdim=True)).clamp(
        min=1e-8
    )  # Avoid division by zero

    dnf = img_pan / denominator  # Match channel dim
    pansharped = dnf * rescaled_ms

    return pansharped
