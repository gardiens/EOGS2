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

import torch
import os
from gaussian_renderer import render
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel
from gaussian_renderer.renderer_cc_shadow import (
    render_resample_virtual_camera,
)
from omegaconf import OmegaConf
from arguments import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)
from scene.MS_scene import MSScene
import cv2


def render_video(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    opt: "OptimizationParams",
    opacity_treshold: float = None,
    output_video_path: str = "output_video",
    ref_idx: int = 0,
):
    with torch.no_grad():
        # load the scene
        gaussians = GaussianModel(dataset.sh_degree)
        scene = MSScene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        iteration = scene.loaded_iter

        bg_color = [
            1,
            0,
            1,
            scene.getTrainCameras()[0].get_msi_cameras().altitude_bounds[0].item(),
            0,
        ]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        device = gaussians._xyz.device
        loaded_viewpoint_cam = torch.load(
            os.path.join(
                dataset.model_path,
                "camera_params",
                f"iteration_{iteration}",
                "camera_params.pth",
            ),
            device,
        )
        # load the ckpt
        for views in [scene.getTestCameras(), scene.getTrainCameras()]:
            for i in range(len(views)):
                for j in range(len(loaded_viewpoint_cam)):
                    if views[i].image_name == loaded_viewpoint_cam[j]["image_name"]:
                        views[i] = loaded_viewpoint_cam[j]["state_dict"]
                        # print("viewpoint_cam",viewpoint_cam[j]["state_dict"])
                        views[i] = views[i].to(gaussians._xyz.device)

        # On génére les images affines comme la caméra nadir

        altitude_min, altitude_max = (
            scene.getTrainCameras()[0]
            .get_pan_cameras()
            .altitude_bounds.cpu()
            .detach()
            .numpy()
            .tolist()
        )

        ###############
        # Update the test color correction if needed
        scene.train_to_test_cc_converter.perform_cc_to_test(
            train_viewpoints=scene.getTrainCameras(),
            test_cameras=scene.getTestCameras(),
            opt=opt,
        )
        test_viewpoint = scene.getTestCameras()
        from utils.camera_utils import get_list_cam

        n = len(get_list_cam(test_viewpoint[0], opt))
        list_images = [[] for k in range(n)]
        print("we have a different of ", n, " cameras to render ")

        #!
        # We were not able to completely incorporate the sun_camera in the metadata , because in some case ( IARPA_PNEWS), some coefficient changed after training, even though the logged configuration said otherwise.
        ref_cam = scene.getTrainCameras()[ref_idx].get_msi_cameras()
        print(
            " WARNING ! Check that in the to_affine and here the idx are coherent, the reference camera name is ",
            ref_cam.image_name,
        )

        sun_camera, _ = ref_cam.get_sun_camera()

        for viewpoint in test_viewpoint:
            for idx, cam in enumerate(get_list_cam(viewpoint, opt)):
                render_pkg = render(cam, gaussians, pipeline, background)["render"]
                raw_render = render_pkg[:3]
                altitude_render = render_pkg[3]

                rendered_uva = torch.stack(cam.UV_grid + (altitude_render,), dim=-1)

                sun_camera_transform, cam2sun = cam.get_sun_camera(
                    f=1
                )  # * we remove the scaling matrice ?

                _, sun_altitude_sample, _, _ = render_resample_virtual_camera(
                    virtual_camera=sun_camera,
                    cam2virt=cam2sun,
                    rendered_uva=rendered_uva,
                    gaussians=gaussians,
                    pipe=pipeline,
                    background=background,
                    return_extra=True,
                )
                sun_altitude_diff = altitude_render - sun_altitude_sample

                renderings = cam.render_pipeline(
                    raw_render=raw_render,
                    sun_altitude_diff=sun_altitude_diff,
                )

                final = renderings["final"].detach().cpu()  # 3xHxW
                if cam.image_type == "pan":
                    # in this case we want to repeat
                    final = final.repeat(3, 1, 1)
                elevation = (
                    altitude_render.detach().cpu().squeeze().repeat(3, 1, 1)
                )  # 3xHxW
                elevation = (elevation - altitude_min) / (altitude_max - altitude_min)
                image = torch.cat([final, elevation], dim=-1)
                list_images[idx].append(image)

        # Create a video given the N images (each has shape 3xHxW)
        for idx in range(len(list_images)):
            images = list_images[idx]
            print("len of images", len(images))
            H, W = images[0].shape[1:]
            out = cv2.VideoWriter(
                filename=f"{output_video_path}_{idx}.avi",
                fourcc=cv2.VideoWriter_fourcc(*"DIVX"),
                fps=30,
                frameSize=(W, H),
                isColor=True,
            )
            for i in range(len(images)):
                img = images[i].permute(1, 2, 0).clip(0, 1).mul(255.0).numpy()  # ?
                img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGB2BGR)
                out.write(img)
            print("Video saved as ", f"{output_video_path}_{idx}.avi")

            out.release()


import hydra


@hydra.main(version_base="1.2", config_path="gs_config", config_name="video.yaml")
def main_hydra_video(cfg: "DictConfig") -> None:
    # Initialize system state (RNG)
    safe_state(cfg.quiet, seed=cfg.seed)
    if not isinstance(cfg.opacity_treshold, type(None)):
        opacity_treshold = float(cfg.opacity_treshold)
    else:
        opacity_treshold = None

    model = ModelParams(cfg.get("model", None))
    pipeline: "PipelineParams" = PipelineParams(cfg=cfg.get("pipeline", None))
    # as we are working with vidoevideo, we change the prefix
    model.source_path = f"{model.source_path}_{cfg.prefix_video}"
    print("final source path is ", model.source_path)
    opt = OptimizationParams(cfg.get("optimization", None))
    render_video(
        dataset=model,
        iteration=cfg.iteration,
        pipeline=pipeline,
        opt=opt,
        output_video_path=cfg.output_video_path,
        # opacity_threshold=opacity_treshold,
    )


if __name__ == "__main__":
    # Set up command line argument parser
    main_hydra_video()
    # parser = ArgumentParser(description="Testing script parameters")
    # model = ModelParams(parser, sentinel=True)
    # pipeline = PipelineParams(parser)
    # parser.add_argument("--iteration", default=-1, type=int)
    # parser.add_argument("--skip_train", action="store_true")
    # parser.add_argument("--skip_test", action="store_true")
    # parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--opacity_treshold", default="")
    # parser.add_argument("--res", default=0.5, type=float)
    # args = get_combined_args(parser)
    # print("Rendering " + args.model_path)

    # # Initialize system state (RNG)
    # safe_state(args.quiet)
    # if args.opacity_treshold != "":
    #     opacity_treshold = float(args.opacity_treshold)
    # else:
    #     opacity_treshold = None

    # render_video(
    #     model.extract(args),
    #     args.iteration,
    #     pipeline.extract(args),
    #     opacity_treshold=opacity_treshold,
    # )
