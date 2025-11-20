#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import pyrootutils

root = str(
    pyrootutils.setup_root(
        search_from=__file__, indicator=".project-root", pythonpath=True, dotenv=True
    )
)
import matplotlib.pyplot as plt
import torch
import os
from tqdm import tqdm
from omegaconf import OmegaConf
from flowmatching.flow_matching import performOpticalmatching
from torchvision import transforms

OmegaConf.register_new_resolver("eval", eval, replace=True)
from arguments import OptimizationParams
from os import makedirs
from gaussian_renderer import render
import iio
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import (
    ModelParams,
    PipelineParams,
    ClearmlParams,
    get_combined_cfg,
)
from gaussian_renderer import GaussianModel

from gaussian_renderer.renderer_cc_shadow import render_resample_virtual_camera
from scene.MS_scene import MSScene

import rasterio
from utils.render_utils import save_img_between_0and1, save_img_notscaled
import numpy as np
from utils.dsm_utils import compute_dsm_from_view
from omegaconf import DictConfig
import hydra

try:
    CLEARML_FOUND = True
    from utils.clearml_utils import safe_resume_clearml, connect_whole
    from utils.clearml_utils import safe_init_clearml

except:
    CLEARML_FOUND = False
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_utils import *  # noqa: F401


class name_setter:
    def __init__(self, return_whole_name: bool = False):
        self.already_seen_name = []
        self.return_whole_name = return_whole_name

    def process(self, viewpoint_cam, name_cam, list_cam, cam):
        image_name = viewpoint_cam.image_name
        if self.return_whole_name:
            final_name = f"{image_name}_{name_cam[list_cam.index(cam)]}"
            return final_name
        # the name is the 5 first letters of the image name
        image_name_int = image_name[:7] if len(image_name) >= 7 else image_name
        final_name = f"{image_name_int}_{name_cam[list_cam.index(cam)]}"
        if final_name in self.already_seen_name:
            self.already_seen_name.append(final_name)

            max_idx = 7 + self.already_seen_name.count(final_name)
            image_name_int = (
                image_name[:max_idx] if len(image_name) >= max_idx else image_name
            )
            final_name = f"{image_name_int}_{name_cam[list_cam.index(cam)]}"
            if final_name in self.already_seen_name:
                final_name = (
                    f"{final_name}_dup{self.already_seen_name.count(final_name)}"
                )
        else:
            self.already_seen_name.append(final_name)
        return final_name


def render_set(
    model_path: "str",
    name: "str",
    iteration: int,
    views: "List[AffineCamera]",
    gaussians: "GaussianModel",
    pipeline: "PipelineParams",
    background: "torch.Tensor",
    scene_params: "list",
    resolution: float,
    msi_pan=None,
    scene_name: str = "default_scene",  # Name of the scene, used for output directory
    sceneparams: "ModelParams" = None,
    opt: "OptimizationParams" = None,
    return_whole_name: bool = False,
):
    if opt.apply_pansharp and opt.load_pan:
        from pansharpening.load_pansharp import load_pansharp

        pansharp_method = load_pansharp(pansharp_cfg=opt.pansharp_cfg)

    if opt.flowmatching.apply_flowmatching:
        warper = performOpticalmatching(
            perform_cst_displacement=opt.flowmatching.perform_cst_displacement,
            mode=opt.flowmatching.mode,
            model_name=opt.flowmatching.model_name,
        )
    # Prepare the output directories
    base_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    # for kind in ["renders", "altitude", "sunpov", "sunpovaltitude", "shadowmap", "shaded", "cc", "sunaltitudesampled", "gt"]:
    for kind in [
        "rawrender",
        "shaded",
        "cc",
        "final",
        "gt",
        "altitude",
        "sunaltitudesampled",
        "sunpovsampled",
        "sun_altitude_diff",
        "shadowmap",
        "sunpov",
        "sunpovaltitude",
        "dsm",
        "accumulated_opacity",
        "nadiraltitudesampled",
        "nadirpovsampled",
        "nadir_altitude_diff",
        "nadirpov",
        "nadirpovaltitude",
        "png",
        # "pan_img",
    ]:
        makedirs(os.path.join(base_path, kind), exist_ok=True)

    # load the viewpoint cams:
    device = views[0].get_pan_cameras().world_view_transform.device
    loaded_viewpoint_cam = torch.load(
        os.path.join(
            model_path,
            "camera_params",
            f"iteration_{iteration}",
            "camera_params.pth",
        ),
        device,
    )
    for i in range(len(views)):
        loaded_cam = False
        for j in range(len(loaded_viewpoint_cam)):
            if views[i].image_name == loaded_viewpoint_cam[j]["image_name"]:
                views[i] = loaded_viewpoint_cam[j]["state_dict"]
                views[i] = views[i].to(gaussians._xyz.device)

    random_i = 0
    get_name = name_setter(return_whole_name=return_whole_name)

    for idx, viewpoint_cam in enumerate(tqdm(views, desc="Rendering progress")):
        # load every camera
        viewpoint_cameras = viewpoint_cam
        list_cam = [viewpoint_cam.get_pan_cameras(), viewpoint_cam.get_msi_cameras()]
        name_cam = ["pan", "msi"]
        for cam in list_cam:
            # get the name
            name = get_name.process(
                viewpoint_cam=viewpoint_cam,
                name_cam=name_cam,
                list_cam=list_cam,
                cam=cam,
            )
            # if the name is Nadir_pan, we change it to Nadir
            if name == "Nadir_pan":
                name = "Nadir"

            name_iio = name + ".iio"

            # render the image
            viewpoint_cam = cam
            gt = viewpoint_cam.original_image[0:3, :, :]
            if sceneparams.repeat_gt and cam.image_type == "pan":
                gt = gt.repeat(3, 1, 1)
            if cam.image_type == "pan" and opt.apply_pansharp and opt.load_pan:
                if not cam.postfix_original_image:
                    gt = pansharp_method(
                        img_pan=gt,
                        img_msi=viewpoint_cameras.get_msi_cameras().original_image,
                    )

            render_pkg = render(viewpoint_cam, gaussians, pipeline, background)[
                "render"
            ]
            raw_render = render_pkg[:3]
            altitude_render = render_pkg[3]
            accumulated_opacity_render = render_pkg[4]

            rendered_uva = torch.stack(
                viewpoint_cam.UV_grid + (altitude_render,), dim=-1
            )

            # Rendering with sun pov
            sun_camera, cam2sun = viewpoint_cam.get_sun_camera()
            sun_rgb_sample, sun_altitude_sample, _, sunpov = (
                render_resample_virtual_camera(
                    virtual_camera=sun_camera,
                    cam2virt=cam2sun,
                    rendered_uva=rendered_uva,
                    gaussians=gaussians,
                    pipe=pipeline,
                    background=background,
                    return_extra=True,
                )
            )
            sun_altitude_diff = altitude_render - sun_altitude_sample

            # Rendering with nadir pov
            nadir_camera, cam2nadir = viewpoint_cam.get_nadir_camera()
            nadir_rgb_sample, nadir_altitude_sample, _, nadirpov = (
                render_resample_virtual_camera(
                    virtual_camera=nadir_camera,
                    cam2virt=cam2nadir,
                    rendered_uva=rendered_uva,
                    gaussians=gaussians,
                    pipe=pipeline,
                    background=background,
                    return_extra=True,
                )
            )
            nadir_altitude_diff = altitude_render - nadir_altitude_sample
            # render with a random camera close to it
            new_camera, camera_to_new = cam.sample_random_camera(
                opt.virtual_camera_extent
            )
            random_rgb_sample, random_altitude_sample, random_uv, random_pov = (
                render_resample_virtual_camera(
                    virtual_camera=new_camera,
                    cam2virt=camera_to_new,
                    rendered_uva=rendered_uva,
                    gaussians=gaussians,
                    pipe=pipeline,
                    background=background,
                    return_extra=True,
                )
            )
            # random_wshad_rgb_sample, random_wshad_altitude_sample, random_wshad_uv, random_wshad_pov = (
            #     render_resample_virtual_camera_wshadowmapping(
            #         virtual_camera=new_camera,
            #         true_cam=cam,
            #         cam2virt=camera_to_new,
            #         rendered_uva=rendered_uva,
            #         gaussians=gaussians,
            #         pipe=pipeline,
            #         background=background,
            #         return_extra=True,
            #     )
            # )
            random_altitude_diff = altitude_render - random_altitude_sample
            random_occlusion_map = (random_altitude_diff.abs() < 0.30) * (
                random_uv.abs() < 1
            ).all(-1)
            random_rgb_occluded = random_rgb_sample * random_occlusion_map

            # random_wshad_altitude_diff=altitude_render - random_wshad_altitude_sample
            # random_wshad_occlusion_map = (random_wshad_altitude_diff.abs() < 0.30) * (
            #     random_wshad_uv.abs() < 1
            # ).all(-1)
            # random_wshad_rgb_occluded = random_wshad_rgb_sample * random_wshad_occlusion_map
            # Render the final image
            renderings = viewpoint_cam.render_pipeline(
                raw_render=raw_render,
                sun_altitude_diff=sun_altitude_diff,
            )
            image = renderings["shaded"]
            if opt.flowmatching.apply_flowmatching:
                if "Nadir" in name:
                    print("we won't apply flow matching on Nadir image")
                else:
                    predicted_flows, gt_image2, image2 = warper.get_and_apply_flow(
                        img_msi_gt=gt,
                        img_msi_target=image,
                    )
                    #! for rendering we never discard the image .
                    gt_flowmatch = gt_image2
                    flow_matched_image = image2
                    if abs(predicted_flows).mean() > 5:
                        print(
                            "for cam image",
                            cam.image_name,
                            cam.image_type,
                            " we have a huge predicted flows, we dont discard it ",
                        )
                    # perform the flow matching on the altitude cam :
                    altitude_flowmatched = warper.apply_flow(
                        img_msi_target=altitude_render, flow=predicted_flows
                    )
            # # Now save the images

            # save img that are not scaled, in this case the .png should be wrong
            # print("psnr between raw_render and gt", psnr(raw_render, gt).mean().double())
            unscaled_features = [
                nadir_altitude_diff,
                nadir_altitude_sample,
                altitude_render,
                # sun_altitude_sample,
                # sun_altitude_diff,
                # sunpov[3],
                accumulated_opacity_render,
                # random_altitude_diff,
                # random_altitude_sample,
                # random_occlusion_map,
                # random_uv,
                # random_wshad_uv
            ]

            unscaled_features_name = [
                "nadir_altitude_diff",
                "nadiraltitudesampled",
                "altitude",
                # "sunaltitudesampled",
                # "sun_altitude_diff",
                # "sunpovaltitude",
                "accumulated_opacity",
                # "random_altitude_diff",
                # "random_altitude_sample",
                # "random_occlusion_map",
                # "random_uv",
                # "random_wshad_uv"
            ]
            if opt.flowmatching.apply_flowmatching:
                unscaled_features.append(altitude_flowmatched)
                unscaled_features_name.append("flowmatched_altitude")
            for i in range(len(unscaled_features_name)):
                name_feature = unscaled_features_name[i]
                makedirs(os.path.join(base_path, name_feature), exist_ok=True)
                save_img_notscaled(
                    name=os.path.join(base_path, name_feature, name),
                    img=unscaled_features[i],
                    save_png=False,
                )

            # save img that are scaled, in this case the .png should be correct
            scaled_feature = [
                nadirpov[:3],
                raw_render,
                sun_rgb_sample,
                nadir_rgb_sample,
                gt,
                # sunpov[:3],
                # random_rgb_sample,
                # random_rgb_occluded,
                # random_wshad_rgb_sample,
                # random_wshad_rgb_occluded,
            ]
            scaled_feature_name = [
                "nadirpov",
                "rawrender",
                "sunpovsampled",
                "nadirpovsampled",
                "gt",
                # "sunpov",
                # "random_rgb_sample",
                # "random_rgb_occluded",
                # "random_shadowed_rgb_sample",
                # "random_shadowed_rgb_sample_occluded",
            ]
            if opt.flowmatching.apply_flowmatching:
                scaled_feature.append(flow_matched_image)
                scaled_feature_name.append("flow_matched_image")
                scaled_feature.append(gt_flowmatch)
                scaled_feature_name.append("gt_flowmatch")

            for i in range(len(scaled_feature_name)):
                name_feature = scaled_feature_name[i]
                makedirs(os.path.join(base_path, name_feature), exist_ok=True)
                save_img_between_0and1(
                    name=os.path.join(base_path, name_feature, name),
                    img=scaled_feature[i],
                    save_png=False,
                )

            for key in renderings:
                makedirs(os.path.join(base_path, key), exist_ok=True)
                out_tmp = renderings[key]
                if isinstance(out_tmp, type(None)):
                    continue
                if len(out_tmp.shape) == 3:
                    out_tmp = out_tmp.permute(1, 2, 0)
                iio.write(os.path.join(base_path, key, name_iio), out_tmp.cpu().numpy())

            # compute the dsm
            profile, dsm = compute_dsm_from_view(
                view=viewpoint_cam,
                rendered_uva=rendered_uva,
                scene_params=scene_params,
                scene_name=scene_name,
            )
            with rasterio.open(
                os.path.join(base_path, "dsm", name_iio), "w", **profile
            ) as f:
                f.write(dsm[:, :, 0], 1)

            if "Nadir" in name:
                dsm_fig = dsm.copy()
                dsm_fig[:, :, 0][np.isnan(dsm_fig[:, :, 0])] = -200
                plt.imshow(dsm_fig[:, :, 0], cmap="gray")
                plt.axis("off")
                plt.tight_layout(pad=0)
                plt.title(
                    "DSM of {} in rendering for scene {}".format(name, scene_name)
                )
                plt.savefig(
                    os.path.join(base_path, "png", name + "_dsm" + ".png"),
                    bbox_inches="tight",
                    pad_inches=0,
                )

                plt.close()

            if idx == random_i:
                final = renderings["final"]

                # Convert tensor to appropriate format for display
                if final.shape[0] == 1:
                    # Single channel - grayscale
                    # For matplotlib, we need to handle this differently
                    image_array = final.squeeze(0).cpu().numpy()  # Shape: (H, W)
                else:
                    # RGB image - permute channels for matplotlib
                    image_array = (
                        final.permute(1, 2, 0).cpu().numpy()
                    )  # Shape: (H, W, 3)

                # Create figure
                fig = plt.figure(figsize=(10, 10))

                # Display with correct colormap
                if final.shape[0] == 1:
                    plt.imshow(image_array, cmap="gray")  # Use grayscale colormap
                else:
                    plt.imshow(image_array)  # Use default RGB colormap

                plt.axis("off")
                plt.title(
                    "Rendered cam image of {} in rendering for scene {}".format(
                        name, scene_name
                    )
                )

                # For saving, use torchvision to handle both cases properly
                to_pil = transforms.ToPILImage()

                if final.shape[0] == 1:
                    # For grayscale, ensure we maintain single channel
                    pil_image = to_pil(
                        final
                    )  # ToPILImage handles single channel correctly
                else:
                    # RGB image
                    pil_image = to_pil(final)

                # Save the image
                pil_image.save(
                    os.path.join(base_path, "png", name + "_rendered" + ".png")
                )
                plt.close()


def render_sets(
    sceneparams: "ModelParams",
    iteration: int,
    pipeline: "PipelineParams",
    skip_train: "bool",
    skip_test: "bool",
    opacity_treshold: float = None,
    resolution: float = 0.5,
    scene_name: str = "default_scene",  # Name of the scene
    opt: "OptimizationParams" = None,
    return_whole_name: bool = False,
):
    with torch.no_grad():
        gaussians = GaussianModel(sceneparams.sh_degree)
        scene = MSScene(sceneparams, gaussians, load_iteration=iteration, shuffle=False)

        if opacity_treshold is not None:
            assert abs(opacity_treshold) <= 1
            if opacity_treshold < 0:
                invalid = gaussians.get_opacity > -opacity_treshold
            else:
                invalid = gaussians.get_opacity < opacity_treshold
            gaussians._opacity[invalid] = -20.0

        bg_color = [
            1,
            0,
            1,
            scene.getTrainCameras()[0].get_msi_cameras().altitude_bounds[0].item(),
            0,
        ]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                model_path=sceneparams.model_path,
                name=f"train_op{opacity_treshold}",
                iteration=scene.loaded_iter,
                views=scene.getTrainCameras(),
                gaussians=gaussians,
                pipeline=pipeline,
                background=background,
                scene_params=[
                    scene.scene_shift,
                    scene.scene_scale,
                    scene.scene_n,
                    scene.scene_l,
                ],
                resolution=resolution,
                scene_name=scene_name,
                sceneparams=sceneparams,
                opt=opt,
                return_whole_name=return_whole_name,
            )

        if not skip_test:
            render_set(
                model_path=sceneparams.model_path,
                name=f"test_op{opacity_treshold}",
                iteration=scene.loaded_iter,
                views=scene.getTestCameras(),
                gaussians=gaussians,
                pipeline=pipeline,
                background=background,
                scene_params=[
                    scene.scene_shift,
                    scene.scene_scale,
                    scene.scene_n,
                    scene.scene_l,
                ],
                resolution=resolution,
                scene_name=scene_name,
                sceneparams=sceneparams,
                opt=opt,
                return_whole_name=return_whole_name,
            )


@hydra.main(version_base="1.2", config_path="gs_config", config_name="rendering.yaml")
def main(cfg: DictConfig) -> None:
    model = ModelParams(cfg.get("model", None))
    pipeline: "PipelineParams" = PipelineParams(cfg=cfg.get("pipeline", None))
    cmlparams = ClearmlParams(cfg.get("clearml", None))
    parser = ArgumentParser(description="Testing script parameters")
    print("resume clearml?", cfg.clearml.resume_clearml)
    opt = OptimizationParams(cfg.get("optimization", None))

    cfg = get_combined_cfg(cfg)
    print("Rendering " + cfg.model.model_path)

    # Initialize system state (RNG)
    safe_state(cfg.quiet, seed=cfg.seed)
    if cfg.opacity_treshold != "" and cfg.opacity_treshold is not None:
        opacity_treshold = float(cfg.opacity_treshold)
    else:
        opacity_treshold = None

    if CLEARML_FOUND and not pipeline.debug:

        if cmlparams.resume_clearml:
            print("we resume?")
            task = safe_resume_clearml(
                project_name=cmlparams.project_name,
                task_name=cmlparams.task_name,
            )
        else:
            task = safe_init_clearml(
                project_name=cmlparams.project_name,
                task_name=cmlparams.task_name,
            )
        # connect args

        connect_whole(
            cfg=cfg,
            task=task,
            name_hyperparams_summary="render config",
            name_connect_cfg="whole render  cfg",
        )

    else:
        print(
            " We didn't find clearml or you are in debug mode, we don't log to Clearml"
        )
    if not cfg.run_rendering:
        print("Rendering is disabled, exiting.")
        return
    scene_name = cfg.rendering_cfg.scene
    render_sets(
        model,
        cfg.iteration,
        pipeline,
        cfg.skip_train,
        cfg.skip_test,
        opacity_treshold=opacity_treshold,
        resolution=cfg.res,
        scene_name=scene_name,
        opt=opt,
        return_whole_name=cfg.return_whole_name,
    )

    if CLEARML_FOUND and not pipeline.debug:
        print("Attempting to close clearml task")
        task.close()
        print("ClearML task closed")


if __name__ == "__main__":
    # Set up command line argument parser
    main()
    from eval.eval_dsm import main_hydra_dsm

    main_hydra_dsm()
    from tsdf import main_hydra_tsdf

    main_hydra_tsdf()
