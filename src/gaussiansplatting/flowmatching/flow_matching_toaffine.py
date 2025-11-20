from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_utils import *
from flowmatching.flow_matching import perform_flow_matching
from gaussian_renderer import render
import torch
from utils.camera_utils import get_list_cam


def adjust_affine(world_view_transform, img_W, img_H, predicted_flows):
    # Calculate horizontal and vertical shifts from the predicted flows
    hor_shift = predicted_flows[0, 0].mean()
    vert_shift = predicted_flows[0, 1].mean()

    # Update the world view transform matrix of the camera
    b = world_view_transform[-1, :]  # Extract the last row

    # Adjust the offsets in the world view transform matrix. The offset is from the gt_image to the image, that's why you have the -.
    # the flowmatch is in the pixel space and affine_proj goes from the world coordinates to space coordinates, so beware of the rescaling
    # idk it's img_W or img_W -1
    b[0] -= hor_shift * 2 / img_W
    b[1] -= vert_shift * 2 / img_H
    world_view_transform[-1, :] = b
    return world_view_transform


def adjust_affine_from_flow(
    scene: "MSScene",
    warper,
    gaussians: "GaussianModel",
    pipe: "PipelineParams",
    bg: "torch.Tensor",
    opt: "OptimizationParams",
):
    """Adjust the affine matrices of the cameras in the scene based on their predicted flows.

    Args:
        scene (MSScene): The scene containing the cameras to be adjusted.
        warper (Any): The warper used for flow matching.
        gaussians (GaussianModel): The Gaussian model used for rendering.
        pipe (PipelineParams): The rendering pipeline.
        bg (torch.Tensor): The background tensor.
        opt (OptimizationParams): Optimization parameters.

    Returns:
        MSScene: The updated scene with adjusted camera affine matrices.
    """
    # Clear CUDA cache to free up memory
    torch.cuda.empty_cache()

    with torch.no_grad():
        # Get a copy of the training cameras from the scene
        viewpoint_stack = scene.getTrainCameras().copy()

        for viewpoint_cam in viewpoint_stack:
            # Retrieve the list of cameras for the current viewpoint
            list_cam = get_list_cam(viewpoint_cam, opt)

            for cam in list_cam:
                # Render the scene from the camera's perspective
                render_pkg = render(cam, gaussians, pipe, bg)
                raw_render = render_pkg["render"][:3]

                # Pass the raw render through the camera's rendering pipeline
                output = cam.render_pipeline(raw_render=raw_render)
                image = output["final"]
                # Get image dimensions
                img_W = image.shape[2]
                img_H = image.shape[1]
                world_view_transform = cam.world_view_transform

                # Perform flow matching to predict the flows
                predicted_flows, _, _ = perform_flow_matching(
                    opt=opt,
                    warper=warper,
                    image=image,
                    gt_image=cam.original_image.cuda(),
                )

                # Set the updated world view transform back to the camera
                cam.world_view_transform = adjust_affine(
                    world_view_transform=world_view_transform,
                    img_W=img_W,
                    img_H=img_H,
                    predicted_flows=predicted_flows,
                )

    # Clear CUDA cache again to free up memory
    torch.cuda.empty_cache()

    return scene
