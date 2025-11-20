from .base_loss import base_Loss
from torch import Tensor
import torch
from utils.loss_utils import ssim


class Translucentshadows_L(base_Loss):
    def __init__(self, w_L_translucentshadows):
        super(Translucentshadows_L, self).__init__()
        self.w_L_translucentshadows = w_L_translucentshadows
        pass

    def forward(self, shadowmap) -> Tensor:
        a = shadowmap
        b = shadowmap.clip(0.05, 0.95)
        L_translucentshadows = -(a * torch.log2(b) + (1 - a) * torch.log2(1 - b)).mean()
        return L_translucentshadows


class photometric_L(base_Loss):
    def __init__(self, lambda_dssim):
        self.lambda_dssim = lambda_dssim
        super(photometric_L, self).__init__()

    def forward(self, image, gt_image, Ll1):
        return (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )


class Suncamera_L(base_Loss):
    def __init__(self, w_L_sun_altitude_resample, w_L_sun_rgb_resample):
        self.w_L_sun_altitude_resample = w_L_sun_altitude_resample
        self.w_L_sun_rgb_resample = w_L_sun_rgb_resample
        super(Suncamera_L, self).__init__()
        pass

    def forward(self, raw_render, sun_rgb_sample, sun_altitude_diff, sun_uv):
        sun_rgb_diff_map = raw_render - sun_rgb_sample
        sun_visibility_map = (sun_altitude_diff > -1e-2) * (sun_uv.abs() < 1).all(-1)
        sun_visibility_map = sun_visibility_map.detach()
        if sun_visibility_map.any():
            L_sun_altitude_resample = (
                sun_altitude_diff.abs() * sun_visibility_map
            ).sum() / (sun_visibility_map.sum())
            L_sun_rgb_resample = (sun_rgb_diff_map.abs() * sun_visibility_map).sum() / (
                sun_visibility_map.sum()
            )
            return L_sun_altitude_resample, L_sun_rgb_resample
