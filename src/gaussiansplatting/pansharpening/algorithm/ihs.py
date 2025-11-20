from .pansharp_utils import torch_interpolation

import torch


def ihs_fusion(img_pan: torch.Tensor, img_msi: torch.Tensor) -> torch.Tensor:
    """Apply the IHS fusion algorithm.

    Parameters:
    - img_pan: tensor of shape (3,H, W)
    - img_msi: tensor of shape (1, h, w)  — assumed RGB

    Returns:
    - fused: tensor of shape (1,H, W)
    """
    # Ensure shapes are correct
    assert img_pan.ndim == 3 and img_pan.shape[0] == 1
    assert img_msi.ndim == 3 and img_msi.shape[0] == 3

    # Add batch dimension to MSI
    img_msi_chw = img_msi.unsqueeze(0)  # [1, 3, h, w]
    img_pan_chw = img_pan.squeeze(0)  # [H, W]

    # Upsample MSI to PAN resolution
    img_msi_up = torch_interpolation(img_msi_chw, img_pan_chw).squeeze(0)  # [3, H, W]

    # Compute intensity
    I0 = img_msi_up.mean(dim=0)  # [H, W]
    Inew = img_pan_chw  # [H, W]

    delta = (Inew - I0).unsqueeze(0)  # [1, H, W]
    fused = img_msi_up + delta  # [3, H, W]

    return fused.clamp(0, 1)
