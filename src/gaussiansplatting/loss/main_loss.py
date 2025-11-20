import torch
from torch import Tensor

from .base_loss import base_Loss
from gaussian_renderer.renderer_cc_shadow import (
    render_resample_virtual_camera,
    render_resample_virtual_camera_wshadowmapping,
)


class myloss(base_Loss):
    def __init__(self, w_L_):
        super(myloss, self).__init__()
        self.w_L_ = w_L_
        pass

    def forward(self, gaussians) -> Tensor:
        pass


class erankLoss(base_Loss):
    def __init__(self, w_L_erank):
        super(erankLoss, self).__init__(weight=w_L_erank)
        pass

    def forward(self, gaussians) -> Tensor:
        s2 = gaussians.get_scaling.square() + 1e-5
        S = s2.sum(dim=1, keepdim=True)
        q = s2 / S
        erankm1 = torch.expm1(-(q * torch.log(q + 1e-6)).sum(dim=1))
        L_erank = (
            torch.log(erankm1 + 1e-5).mul(-1).clip(min=0.0) + s2.amin(1).sqrt()
        ).mean()
        return L_erank

    def get_loss_name(self):
        return "L_erank"


class Total_variation(base_Loss):
    def __init__(self, w_L_TV_altitude):
        super(Total_variation, self).__init__(weight=w_L_TV_altitude)

        pass

    def forward(self, altitude_render) -> Tensor:
        diff1 = altitude_render[..., 1:, :] - altitude_render[..., :-1, :]
        diff2 = altitude_render[..., :, 1:] - altitude_render[..., :, :-1]
        L_TV_altitude = 0.5 * (diff1.abs().mean() + diff2.abs().mean())
        return L_TV_altitude

    def get_loss_name(self):
        return "L_TV_altitude"


class RandomcamRendering_Loss(base_Loss):
    def __init__(
        self,
        w_L_new_altitude_resample,
        w_L_new_rgb_resample,
        render_type,
        use_gt: bool = False,
    ):
        """Small deformation loss to enforce consistent rendering under small camera perturbation.

        render_type: "rawrender" or "rawrender_wshadow". Decide on which rendering to use for the random camera.
        use_gt: if True, use the gt image for the random camera loss instead of the initial raw_render or shaded image.
        Beware REGISTRATION ISSUE !
        """
        super(RandomcamRendering_Loss, self).__init__(weight=0)
        self.w_L_randomcam_rendering = w_L_new_altitude_resample
        self.w_L_new_rgb_resample = w_L_new_rgb_resample
        self.render_type = render_type
        self.use_gt = use_gt
        if self.use_gt:
            print(
                "You are using gt image for random camera loss, be sure this is what you want"
            )
        assert self.render_type in ["rawrender", "rawrender_wshadow"], (
            "render type not implemented,got " + self.render_type
        )

    def _forward(
        self, new_occlusion_map, new_altitude_diff, new_rgb_diff_map
    ) -> Tensor:
        if new_occlusion_map.any():
            L_new_altitude_resample = (
                new_altitude_diff.abs() * new_occlusion_map
            ).sum() / (new_occlusion_map.sum())
            L_new_rgb_resample = (new_rgb_diff_map.abs() * new_occlusion_map).sum() / (
                new_occlusion_map.sum()
            )
            return L_new_altitude_resample, L_new_rgb_resample
        else:
            return 0, 0

    def forward(
        self,
        new_camera,
        true_cam,
        camera_to_new,
        rendered_uva,
        gaussians,
        pipe,
        bg,
        altitude_render,
        raw_render,
        gt_image,
        shaded,
    ):
        """
        new_camera: the random camera
        true_cam: the original true camera (used to get raw_render and shaded)
        camera_to_new: the transformation from true camera to new camera
        raw_render: the raw render from the true camera
        shaded: the image obtained with the camera pipeline of the true camera
        gt_imge: the gt_image"""

        if self.render_type == "rawrender":
            new_rgb_sample, new_altitude_sample, new_uv = self._rawrender_from_camera(
                new_camera=new_camera,
                camera_to_new=camera_to_new,
                rendered_uva=rendered_uva,
                gaussians=gaussians,
                pipe=pipe,
                bg=bg,
            )
            rgb_render = raw_render
        elif self.render_type == "rawrender_wshadow":
            new_rgb_sample, new_altitude_sample, new_uv = (
                self._rawrender_from_camera_wshadow(
                    new_camera=new_camera,
                    true_cam=true_cam,
                    camera_to_new=camera_to_new,
                    rendered_uva=rendered_uva,
                    gaussians=gaussians,
                    pipe=pipe,
                    bg=bg,
                )
            )
            rgb_render = shaded
        else:
            raise NotImplementedError(
                "render type not implemented,got ", self.render_type
            )
        if self.use_gt:
            rgb_render = gt_image  #! Beware registration issue with flowmatching
        # actual logic of the code.
        new_altitude_diff = altitude_render - new_altitude_sample
        new_rgb_diff_map = rgb_render - new_rgb_sample
        new_occlusion_map = (new_altitude_diff.abs() < 0.30) * (new_uv.abs() < 1).all(
            -1
        )
        new_occlusion_map = new_occlusion_map.detach()
        if new_occlusion_map.any():
            L_new_altitude_resample, L_new_rgb_resample = self._forward(
                new_occlusion_map=new_occlusion_map,
                new_altitude_diff=new_altitude_diff,
                new_rgb_diff_map=new_rgb_diff_map,
            )
        else:
            L_new_altitude_resample = torch.tensor(0.0, device=camera_to_new.device)
            L_new_rgb_resample = torch.tensor(0.0, device=camera_to_new.device)
        return L_new_altitude_resample, L_new_rgb_resample

    def _forward_from_camera(
        self,
        new_camera,
        camera_to_new,
        rendered_uva,
        gaussians,
        pipe,
        bg,
        altitude_render,
        raw_render,
    ):
        new_rgb_sample, new_altitude_sample, new_uv = render_resample_virtual_camera(
            virtual_camera=new_camera,
            cam2virt=camera_to_new,
            rendered_uva=rendered_uva,
            gaussians=gaussians,
            pipe=pipe,
            background=bg,
        )
        new_altitude_diff = altitude_render - new_altitude_sample
        new_rgb_diff_map = raw_render - new_rgb_sample
        new_occlusion_map = (new_altitude_diff.abs() < 0.30) * (new_uv.abs() < 1).all(
            -1
        )
        new_occlusion_map = new_occlusion_map.detach()
        return self.forward(new_occlusion_map, new_altitude_diff, new_rgb_diff_map)

    def _rawrender_from_camera(
        self, new_camera, camera_to_new, rendered_uva, gaussians, pipe, bg
    ):
        new_rgb_sample, new_altitude_sample, new_uv = render_resample_virtual_camera(
            virtual_camera=new_camera,
            cam2virt=camera_to_new,
            rendered_uva=rendered_uva,
            gaussians=gaussians,
            pipe=pipe,
            background=bg,
        )
        return new_rgb_sample, new_altitude_sample, new_uv

    def _rawrender_from_camera_wshadow(
        self, new_camera, true_cam, camera_to_new, rendered_uva, gaussians, pipe, bg
    ):
        virtual_rgb_sample, virtual_altitude_sample, virtual_uv = (
            render_resample_virtual_camera_wshadowmapping(
                virtual_camera=new_camera,
                true_cam=true_cam,
                cam2virt=camera_to_new,
                rendered_uva=rendered_uva,
                gaussians=gaussians,
                pipe=pipe,
                background=bg,
                return_extra=False,
            )
        )
        return virtual_rgb_sample, virtual_altitude_sample, virtual_uv

    def log_loss(
        self,
        tb_writer: "SummaryWriter",
        L_new_altitude_resample,
        L_new_rgb_resample,
        iteration: int,
    ) -> None:
        tb_writer.add_scalar(
            f"loss/L_new_altitude_resample", L_new_altitude_resample, iteration
        )
        tb_writer.add_scalar(f"loss/L_new_rgb_resample", L_new_rgb_resample, iteration)
