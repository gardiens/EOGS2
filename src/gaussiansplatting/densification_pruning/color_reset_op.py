# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from omegaconf import OmegaConf

# add the regiser eval
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)
import torch
from utils.sh_utils import RGB2SH

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
try:
    CLEARML_FOUND = True
    print("ClearML found")
except:
    CLEARML_FOUND = False

from loss import *

from gaussian_renderer.renderer_cc_shadow import (
    render_all_views,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_utils import *


def color_reset(gaussians, scene, pipe):
    to_reset = torch.full((gaussians._xyz.size(0),), False, device="cuda", dtype=bool)
    myoutput = render_all_views(scene.getTrainCameras(), gaussians, pipe)
    for myoutputview in myoutput:
        shadowmap = myoutputview["shadow"]
        pts = myoutputview["projxyz"]
        to_reset_iter = (
            1
            - torch.max_pool2d(
                1 - shadowmap[None, None], 5, stride=1, padding=2
            ).squeeze()
        )
        to_reset_iter = (
            torch.nn.functional.grid_sample(
                to_reset_iter[None, None],
                pts[None, None],
                mode="bilinear",
                align_corners=True,
                padding_mode="zeros",
            ).squeeze()
            < 0.5
        )
        to_reset = to_reset | to_reset_iter

    with torch.no_grad():
        gaussians._opacity[to_reset] = gaussians.inverse_opacity_activation(
            0.005 * torch.ones_like(gaussians._opacity[to_reset])
        )
        gaussians._features_dc[to_reset] = RGB2SH(
            torch.full_like(gaussians._features_dc[to_reset], 1.1)
        )
        gaussians._scaling[to_reset] = gaussians.scaling_inverse_activation(
            (1.0 / 400) * torch.ones_like(gaussians._scaling[to_reset])
        )  # TODO: the scaling should be reset using kNN

        for param in [
            gaussians._opacity,
            gaussians._features_dc,
            gaussians._scaling,
            # gaussians._rotation,
            # gaussians._xyz,
        ]:
            mask = to_reset.squeeze().clone()
            while len(mask.shape) < len(param.shape):
                mask = mask.unsqueeze(-1)
            gaussians.optimizer.state[param]["exp_avg"].masked_fill_(mask, 0.0)
            gaussians.optimizer.state[param]["exp_avg_sq"].masked_fill_(mask, 0.0)
