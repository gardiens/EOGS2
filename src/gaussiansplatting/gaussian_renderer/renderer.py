import torch
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.gaussian_model import GaussianModel
from utils.sh_utils import SH2RGB
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_utils import *


def render(
    viewpoint_camera: "AffineCamera",
    pc: GaussianModel,
    pipe: "PipelineParams",
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    separate_sh=False,
    override_color=None,
    use_trained_exp=False,
):
    """Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    viewmatrix = viewpoint_camera.world_view_transform
    projmatrix = viewpoint_camera.full_proj_transform
    if viewpoint_camera.learn_wv_only_lastparam:
        # don't forget clone otherwise it's a pointer ;)
        viewmatrix = viewpoint_camera.world_view_transform.clone()
        projmatrix = viewpoint_camera.full_proj_transform.clone()
        # we add to the last row of the world to view transform
        viewmatrix[-1, :] = viewmatrix[-1, :] + viewpoint_camera.last_row
        projmatrix[-1, :] = projmatrix[-1, :] + viewpoint_camera.last_row
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # We recompiled the rasterizer to render RGB (zero degree) and altitude at the same time
    if override_color is None:
        rgb = SH2RGB(pc._features_dc).squeeze(1)
        xyz_uva = viewpoint_camera.ECEF_to_UVA(pc._xyz)
        altitude = xyz_uva[..., 2].unsqueeze(-1)
        constant = torch.ones_like(altitude)
        colors_precomp = torch.cat([rgb, altitude, constant], dim=-1)
    else:
        colors_precomp = override_color
    shs = None
    assert bg_color.shape[-1] == colors_precomp.shape[-1]

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # depth_image
    rendered_image, radii, invdepths = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    # new update:
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = (
            torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(
                2, 0, 1
            )
            + exposure[:3, 3, None, None]
        )
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        # "depth": depth_image,
    }
    if pipe.require_radii:
        out["visibility_filter"] = (radii > 0).nonzero()
        out["radii"] = radii

    return out
