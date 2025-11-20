import torch
from .renderer import render


# added by previosu code
def render_resample_virtual_camera(
    virtual_camera,
    cam2virt,
    rendered_uva,
    gaussians,
    pipe,
    background,
    return_extra=False,
):
    """
    INPUTS:
        - virtual_camera: the camera from which we render the scene
        - cam2virt: the transformation matrix from the actual camera to the virtual camera (UVA->UVA)
        - rendered_uva: the UVA meshgrid (H,W,3) of the rendered image from the actual camera
        - gaussians, pipe, background: the usual suspects
    OUTPUTS:
        - virtual_rgb_sample: the RGB image of the virtual camera resampled to the actual camera
        - virtual_altitude_sample: the altitude image of the virtual camera resampled to the actual camera
        - virtual_uv: the UVA coordinates where the virtual image was sampled
    """
    # Step 1. Render the scene from the virtual camera pov
    # the virtual_render_pkg is now a (H,W,3+1) tensor with channels RGB+altitude
    virtual_render_pkg = render(virtual_camera, gaussians, pipe, background)
    virtual_render = virtual_render_pkg["render"]

    # Step 2. Use the coordinate transformation to reproject the actual image to the virtual image
    virtual_uv = torch.einsum("...ij,...j->...i", cam2virt, rendered_uva)[
        ..., :2
    ]  # This operation perform : uv= cam2virt @ uva pixel per pixel .

    # Step 3. Sample the virtual image at the reprojected coordinates
    virtual_render_sample = torch.nn.functional.grid_sample(
        virtual_render.unsqueeze(0),
        virtual_uv.unsqueeze(0),
        align_corners=True,
    ).squeeze(0)  # (H,W,3+1)

    virtual_rgb_sample = virtual_render_sample[:3]
    virtual_altitude_sample = virtual_render_sample[3]

    virtual_altitude_sample[(virtual_uv.abs() > 1).any(-1)] = -100
    if return_extra:
        return (
            virtual_rgb_sample,
            virtual_altitude_sample,
            virtual_uv,
            virtual_render,
        )
    return virtual_rgb_sample, virtual_altitude_sample, virtual_uv


def render_resample_virtual_camera_wshadowmapping(
    virtual_camera,
    true_cam,
    cam2virt,
    rendered_uva,
    gaussians,
    pipe,
    background,
    return_extra=False,
):
    """Render the scene from the virtual camera and we apply here the whole true_cam pipeline .

    INPUTS:
        - virtual_camera: the camera from which we render the scene
        - cam2virt: the transformation matrix from the actual camera to the virtual camera (UVA->UVA)
        - rendered_uva: the UVA meshgrid (H,W,3) of the rendered image from the actual camera
        - gaussians, pipe, background: the usual suspects
    OUTPUTS:
        - virtual_rgb_sample: the RGB image of the virtual camera resampled to the actual camera
        - virtual_altitude_sample: the altitude image of the virtual camera resampled to the actual camera
        - virtual_uv: the UVA coordinates where the virtual image was sampled
    """
    # Step 1. Render the scene from the virtual camera pov
    # the virtual_render_pkg is now a (H,W,3+1) tensor with channels RGB+altitude
    virtual_render_pkg = render(virtual_camera, gaussians, pipe, background)
    virtual_render = virtual_render_pkg["render"]
    virtual_raw_render = virtual_render_pkg["render"][:3]
    virtual_altitude = virtual_render_pkg["render"][3]
    virtual_uva = torch.einsum("...ij,...j->...i", cam2virt, rendered_uva)[
        ..., :
    ]  # This operation perform : uv= cam2virt @ uva pixel per pixel .

    # plt.imshow(virtual_raw_render.detach().cpu().numpy().transpose(1,2,0))
    # plt.title("virtual raw render")
    # plt.show()
    # perform the virt_to_sun transformation
    sun_camera, camera_to_sun = true_cam.get_sun_camera()
    cam2virt_inv = torch.inverse(cam2virt)
    virt_to_sun = torch.matmul(cam2virt_inv, camera_to_sun)

    # now render the sun view
    sun_rgb_sample, sun_altitude_sample, _ = render_resample_virtual_camera(
        virtual_camera=sun_camera,
        cam2virt=virt_to_sun,
        rendered_uva=virtual_uva,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
    )
    # plt.imshow(sun_rgb_sample.detach().cpu().numpy().transpose(1,2,0))
    # plt.title("virtual sun render")
    # plt.show()
    virtual_altitude_diff = virtual_altitude - sun_altitude_sample

    # * Specificity here: we render the virtual camera with the whole pipeline.

    true_output = true_cam.render_pipeline(
        raw_render=virtual_raw_render,
        sun_altitude_diff=virtual_altitude_diff,
    )
    # plt.imshow(true_output["shaded"].detach().cpu().numpy().transpose(1,2,0))
    # plt.title("virtual shaded render")
    # plt.show()

    virtual_render = torch.cat([true_output["shaded"], virtual_render[3:]], dim=0)
    n = true_output["shaded"].shape[0]

    # Step 2. Use the coordinate transformation to reproject the actual image to the virtual image
    virtual_uv = virtual_uva[..., :2]

    # Step 3. Sample the virtual image at the reprojected coordinates
    virtual_render_sample = torch.nn.functional.grid_sample(
        virtual_render.unsqueeze(0),
        virtual_uv.unsqueeze(0),
        align_corners=True,
    ).squeeze(0)  # (H,W,3+1)

    virtual_rgb_sample = virtual_render_sample[:n]
    virtual_altitude_sample = virtual_render_sample[n]

    virtual_altitude_sample[(virtual_uv.abs() > 1).any(-1)] = -100
    if return_extra:
        return (
            virtual_rgb_sample,
            virtual_altitude_sample,
            virtual_uv,
            virtual_render_sample,
        )
    return virtual_rgb_sample, virtual_altitude_sample, virtual_uv


@torch.no_grad()
def render_all_views(cameras, gaussians, pipe, bg=None, override_color=None):
    if bg is None:
        bg = torch.rand((5), device="cuda")

    out = []

    for viewpoint_cam in cameras:
        bg[3] = viewpoint_cam.altitude_bounds[0].item()
        bg[4] = 0.0
        render_pkg = render(
            viewpoint_cam, gaussians, pipe, bg, override_color=override_color
        )
        raw_render = render_pkg["render"][:3]
        altitude_render = render_pkg["render"][3]
        rendered_uva = torch.stack(viewpoint_cam.UV_grid + (altitude_render,), dim=-1)

        sun_camera, camera_to_sun = viewpoint_cam.get_sun_camera()
        _, sun_altitude_sample, _ = render_resample_virtual_camera(
            virtual_camera=sun_camera,
            cam2virt=camera_to_sun,
            rendered_uva=rendered_uva,
            gaussians=gaussians,
            pipe=pipe,
            background=bg,
        )
        sun_altitude_diff = altitude_render - sun_altitude_sample

        output = viewpoint_cam.render_pipeline(
            raw_render=raw_render,
            sun_altitude_diff=sun_altitude_diff,
        )

        out.append(
            {
                "image_name": viewpoint_cam.image_name,
                "shadow": output["shadowmap"],
                "raw_render": raw_render,
                "cc": output["cc"],
                "render": output["final"],
                "projxyz": viewpoint_cam.ECEF_to_UVA(gaussians.get_xyz)[:, :2],
                "altitude_render": altitude_render,
            }
        )

    return out
