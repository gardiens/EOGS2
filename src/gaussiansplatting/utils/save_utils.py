import torch

from .sh_utils import SH2RGB, RGB2SH
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_utils import *


def normalize_before_saving(scene: "MSScene", gaussians: "GaussianModel"):
    view = None
    for c in scene.getTrainCameras():
        if c.is_reference_camera:
            view = c.get_msi_cameras()
            break
    assert view is not None

    A1 = view.color_correction.weight.squeeze()
    b1 = view.color_correction.bias.squeeze()
    A1inv = torch.linalg.inv(A1.double()).float()

    rgb_colors = SH2RGB(gaussians._features_dc)
    normalized_rgb_colors = torch.einsum("ij,...j->...i", A1, rgb_colors) + b1
    gaussians._features_dc = RGB2SH(normalized_rgb_colors)

    for c in scene.getTrainCameras():
        c = c.get_msi_cameras()
        Ai = c.color_correction.weight.squeeze()
        bi = c.color_correction.bias.squeeze()
        AiA1inv = Ai @ A1inv
        AiAiinvb1 = Ai @ A1inv @ b1
        c.color_correction.weight.data = AiA1inv.reshape(3, 3, 1, 1)
        c.color_correction.bias.data = -AiAiinvb1 + bi
