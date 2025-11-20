from loss.base_loss import base_Loss
from pansharpening.load_pansharp import load_pansharp

import torch


class Pansharploss(base_Loss):
    def __init__(self, lambda_L_pansharp, pansharp_cfg: "DictConfig"):
        super(Pansharploss, self).__init__()
        self.lambda_L_pansharp = lambda_L_pansharp
        self.pansharp_method = load_pansharp(pansharp_cfg=pansharp_cfg)
        print("the pansharp_method is", self.pansharp_method)

    def forward(self, syn_msi_image, gt_pan_image, gt_msi_image):
        # Compute the gradient of the PAN image
        # Apply the pansharpening method*

        pansharped_image = self.pansharp_method(
            img_pan=gt_pan_image, img_msi=gt_msi_image
        )
        # Compute the L2 loss between the pansharpened gt image and the synthetized image
        l_pansharp = torch.mean((syn_msi_image - pansharped_image) ** 2)
        return l_pansharp
