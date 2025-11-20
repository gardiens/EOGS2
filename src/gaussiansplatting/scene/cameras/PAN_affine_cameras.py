from scene.msi_to_pan.transf_msi_to_pan import load_msi_to_pan
from scene.cameras.affine_cameras import AffineCamera
import typing

if typing.TYPE_CHECKING:
    from utils.typing_utils import *
import torch
from torch import nn


class PANAffineCamera(AffineCamera):
    def __init__(
        self,
        caminfo: "AffineCameraInfo",
        image,
        gt_alpha_mask,
        data_device="cuda",
        is_reference_camera=False,
        image_type="pan",
        args: "ModelParams" = None,
    ):
        super(PANAffineCamera, self).__init__(
            caminfo=caminfo,
            image=image,
            gt_alpha_mask=gt_alpha_mask,
            data_device=data_device,
            is_reference_camera=is_reference_camera,
            image_type=image_type,
            args=args,
        )
        self.image_type = image_type
        self.mode = "pan"
        # if self.use_shadow:
        assert (
            args is not None
        ), "PANAffineCamera requires args to be provided for msi_to_pan conversion, got args None"
        self.msi_to_pan = load_msi_to_pan(args.msi_to_pan).to(self.data_device)

        # override the original image in case of PAN camera
        self.postfix_original_image = False
        # self.original_image = self.compute_original_image(image=image,cfg_original_image=args.pansharpening_cfg)
        if args.weird_pan_setup:
            self.weird_pan_setup = True
            print("WE GONNA DO COLOR CORRECTION AFTER THE MSI TO PAN")

            # override parent's (3,3) conv
            self.color_correction = nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=1, bias=True
            ).to(self.data_device)
            with torch.no_grad():
                self.color_correction.weight = nn.Parameter(
                    torch.eye(1, device=self.data_device).reshape(1, 1, 1, 1)
                )
                self.color_correction.bias = nn.Parameter(
                    torch.zeros(1, device=self.data_device)
                )

            self.inshadow_color_correction = nn.Parameter(
                torch.zeros(1, device=self.data_device).reshape(1, 1, 1) + 0.05
            )

        else:
            self.weird_pan_setup = False

    def postfix_compute_original_image(self, pan_image, msi_image, pansharp_method):
        # technical things if you need  to compute the original image because you may need to have the MSI image for pansharpening
        gt_image = pansharp_method(img_pan=gt_image, img_msi=msi_image)
        gt_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.original_image = gt_image
        self.postfix_original_image = True
        return gt_image

    def render_pipeline(self, raw_render, sun_altitude_diff=None):
        if not self.weird_pan_setup:
            return self._render_pipeline(
                raw_render=raw_render, sun_altitude_diff=sun_altitude_diff
            )
        else:
            return self._render_pipeline_weird(
                raw_render=raw_render, sun_altitude_diff=sun_altitude_diff
            )

    def _render_pipeline(self, raw_render, sun_altitude_diff=None):
        # raw_render : (3,H,W)
        # sun_altitude_diff : (H,W) or None

        # TODO: harmonize PAN and affine camera to use the code only fromAffineCamera
        render = raw_render.unsqueeze(0)
        # print("the render before cc is",render)
        # # Step 2: Apply color correction

        # Step 2: Apply color correction
        if self.use_cc:  # not self.is_reference_camera:
            cc = self.color_correction(render)
        elif self.use_exposure:
            exposure = self.exposure
            B, C, H, W = render.shape

            # flatten pixels -> (B, 3, H*W)
            render_flat = render.view(B, C, -1)

            # affine transform: (B, 3, 3) @ (B, 3, N) + bias
            cc = torch.bmm(exposure[:, :, :3], render_flat) + exposure[:, :, 3:4]

            # reshape back to image
            cc = cc.view(B, C, H, W)

        else:
            cc = render
        # print("the render after cc is",cc)
        # print("pixel 0 value after cc",cc[0,:,0,0])

        # Step 1: Apply shades
        if self.use_shadow and sun_altitude_diff is not None:
            shadow = self.shadow_map(
                sun_altitude_diff
            )  # Now is between 0 and 1, dimensions are (H,W)
            shaded = shadow * cc + (1 - shadow) * self.inshadow_color_correction * cc
        else:
            shadow = None
            shaded = cc

        final = shaded
        # Transform the PAN image if needed
        if self.mode == "pan":
            # print("the msi_to_pan is", self.msi_to_pan)
            # print("shaded shape is", shaded.shape)
            # print("pixel 0 value",shaded[0,:,0,0])
            shaded = self.msi_to_pan(shaded)
            # print("after the pixel 0 is ",shaded[0,:,0,0])
        else:
            raise ValueError(
                f"Unknown mode {self.mode} for PANAffineCamera. Expected 'pan'."
            )
        final = shaded
        # paint transient on top.
        # if self.use_transient:
        #     mask = self.transient_mask.clip(0, 1)
        #     final = final * (1 - mask) + self.transient_canvas * mask
        # final = shaded * (1-mask) + self.original_image * mask
        return {
            "shadowmap": shadow,
            "shaded": shaded.squeeze(0),
            "cc": cc.squeeze(0),
            "final": final.squeeze(0),
        }

    def _render_pipeline_weird(self, raw_render, sun_altitude_diff=None):
        render = raw_render.unsqueeze(0)
        if self.mode == "pan":
            shaded = self.msi_to_pan(render)
        else:
            raise ValueError(
                f"Unknown mode {self.mode} for PANAffineCamera. Expected 'pan'."
            )
        if True:  # not self.is_reference_camera:
            cc = self.color_correction(shaded)
        final = cc

        if self.use_shadow and sun_altitude_diff is not None:
            shadow = self.shadow_map(
                sun_altitude_diff
            )  # Now is between 0 and 1, dimensions are (H,W)
            shaded = shadow * cc + (1 - shadow) * self.inshadow_color_correction * cc
        else:
            shadow = None
            shaded = shaded

        final = shaded

        return {
            "shadowmap": shadow,
            "shaded": shaded.squeeze(0),
            "cc": cc.squeeze(0),
            "final": final.squeeze(0),
        }

    def unfreeze_msi_to_pan(self):
        # Unfreeze the msi_to_pan parameters
        for param in self.msi_to_pan.parameters():
            param.requires_grad = True
        self.msi_to_pan.learn_conv2d = True
