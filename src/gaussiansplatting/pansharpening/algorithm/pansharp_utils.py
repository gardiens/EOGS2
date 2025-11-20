import torch
import torch.nn.functional as F


def torch_interpolation(
    img_msi_dw: torch.Tensor, img_pan_up: torch.Tensor
) -> torch.Tensor:
    return F.interpolate(
        img_msi_dw, size=img_pan_up.shape, mode="bilinear", align_corners=False
    )


def resize_tensor(img_tensor, size_out):
    # img_tensor shape: (C, H, W)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dim: (1, C, H, W)
    resized = F.interpolate(
        img_tensor, mode="bicubic", align_corners=False, size=size_out
    )
    return resized.squeeze(0)  # Remove batch dim
