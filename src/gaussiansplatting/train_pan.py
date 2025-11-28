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
import os
import uuid
from random import randint
from omegaconf import DictConfig, OmegaConf
import hydra
from pansharpening.load_pansharp import load_pansharp

# add the regiser eval
from omegaconf import OmegaConf
import numpy as np
from flowmatching.flow_matching import performOpticalmatching, perform_flow_matching
from flowmatching.flow_matching_toaffine import adjust_affine_from_flow
from utils.callback_utils import early_stopping

OmegaConf.register_new_resolver("eval", eval, replace=True)
import torch
from eval.eval_dsm import Mae_Computer
from arguments import (
    ClearmlParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    loggingGS,
)
from gaussian_renderer import render
from scene import GaussianModel
from scene.MS_scene import MSScene
from utils.image_utils import psnr
from utils.save_utils import normalize_before_saving
from tqdm import tqdm
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss
from utils.config_io import recursive_args_from_pg
from utils.dsm_utils import compute_dsm_from_view
from loss.PAN_loss import (
    Lgradient_pan,
)
from densification_pruning.color_reset_op import color_reset

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
try:
    import clearml
    CLEARML_FOUND = True
    print("ClearML found")
except:
    CLEARML_FOUND = False

from scene.cameras import MS_affine_cameras
from loss import *

from gaussian_renderer.renderer_cc_shadow import (
    render_resample_virtual_camera,
    render_all_views,
)
import rasterio
from typing import TYPE_CHECKING
from utils.camera_utils import get_list_cam

if TYPE_CHECKING:
    from typing import List
    from utils.typing_utils import *


def unfreeze_wv_transform(scene, opt):
    for viewpoint_cam in scene.getTrainCameras():
        list_cam: "List[AffineCamera]" = get_list_cam(viewpoint_cam, opt)
        for cam in list_cam:
            if cam.learn_wv_only_lastparam:
                cam.last_row.requires_grad = True
            else:
                cam.world_view_transform.requires_grad = True
    return


def training(
    sceneparams: "ModelParams",
    opt: "OptimizationParams",
    pipe: "PipelineParams",
    GS_loger: "loggingGS",
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    start_checkpoint,
    debug_from,
):
    print("the gt_dir is", GS_loger.dsm.gt_dir)
    mae_computer = Mae_Computer(
        gt_dir=GS_loger.dsm.gt_dir,
        aoi_id=GS_loger.dsm.aoi_id,
        enable_vis_mask=GS_loger.dsm.enable_vis_mask,
        filter_tree=GS_loger.dsm.filter_tree,
    )
    first_iter = 0
    tb_writer = prepare_output_and_logger(sceneparams)

    gaussians = GaussianModel(sceneparams.sh_degree)
    scene = MSScene(args=sceneparams, gaussians=gaussians)

    gaussians.training_setup(opt)
    if start_checkpoint:
        (model_params, first_iter) = torch.load(start_checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1, 1, 1] if sceneparams.white_background else [0, 0, 0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_Lphotometric_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    camera_optimizer = torch.optim.Adam(
        [{"params": cam.parameters()} for cam in scene.getTrainCameras()],
        lr=opt.camera_lr,
    )

    # at the beginning freeze the msi_to_pan params
    if opt.freeze_start_msitopan_params:
        for cam in scene.getTrainCameras():
            pan_cam = cam.get_pan_cameras()
            for param in pan_cam.msi_to_pan.parameters():
                param.requires_grad = False
    if sceneparams.repeat_gt:
        print("we will repeat the GT images 3 times to have 3 channels")
    if opt.early_stopping.use_early_stopping:
        print("WE WILL USE THE EARLY STOPPING")
        early_stopper = early_stopping(
            patience=opt.early_stopping.patience,
            operator=opt.early_stopping.operator,
            metric_name=opt.early_stopping.metric_name,
        )
    init_number_of_gaussians = len(gaussians._xyz)
    #! Initialize all the losses computer
    opacity_l = OpacityLoss(
        w_L_opacity=opt.w_L_opacity, init_number_of_gaussians=init_number_of_gaussians
    )
    accumulatedopacity_l = AccumulatedOpacity(
        w_L_accumulated_opacity=opt.w_L_accumulated_opacity
    )
    TV_l = Total_variation(w_L_TV_altitude=opt.w_L_TV_altitude)
    erank_l = erankLoss(w_L_erank=opt.w_L_erank)
    translucentshadows_l = Translucentshadows_L(
        w_L_translucentshadows=opt.w_L_translucentshadows
    )
    photometric_l = photometric_L(lambda_dssim=opt.lambda_dssim)
    suncamera_l = Suncamera_L(
        w_L_sun_altitude_resample=opt.w_L_sun_altitude_resample,
        w_L_sun_rgb_resample=opt.w_L_sun_rgb_resample,
    )
    opacity_radii_l = radiiOpacityLoss(
        w_L_opacity=opt.w_L_opacity, init_number_of_gaussians=init_number_of_gaussians
    )
    flowmatch_l = flowmatchLoss(w_L_flowmatch=opt.w_L_flowmatch)
    assert not (
        opt.w_L_opacity_radii > 0 and opt.w_L_opacity > 0
    ), f"You cannot have both opacity and opacity_radii loss activated at the same time, please set one of the two weights to zero, got w_L_opacity={opt.w_L_opacity} and w_L_opacity_radii={opt.w_L_opacity_radii}"
    randomcamera_l = RandomcamRendering_Loss(
        w_L_new_altitude_resample=opt.w_L_new_altitude_resample,
        w_L_new_rgb_resample=opt.w_L_new_rgb_resample,
        render_type=opt.random_camera.randomcamera_render_type,
        use_gt=opt.random_camera.use_gt,
    )
    if opt.random_camera.use_gt and opt.flowmatching.apply_flowmatching:
        raise ValueError(
            "You cannot use gt with random camera when you also use flowmatching, are you really sure about it? The random camera will be slightly off w.r.t the gt image"
        )

    pan_l = photometric_L(lambda_dssim=opt.lambda_dssim)
    gradient_pan_l = Lgradient_pan(lambda_lgradient_pan=opt.w_Lgradient_pan)

    # initialize metric
    print("WE ARE LOIADING:", "msi:", opt.load_msi, "pan:", opt.load_pan)
    mae = np.inf
    mae_wtree = np.inf
    pan_ssim = 0
    pan_psnr = 0
    msi_psnr = 0
    msi_ssim = 0
    n_pan = 0
    n_msi = 0
    metric_dict = {
        "mae": mae,
        "mae_wtree": mae_wtree,
        "pan_ssim": pan_ssim,
        "pan_psnr": pan_psnr,
        "msi_psnr": msi_psnr,
        "msi_ssim": msi_ssim,
        "photometric": 0.0,
        "L1": 0.0,
        "n_photo": 0,
    }
    # pansharpening if needed
    if opt.apply_pansharp and opt.load_pan:
        print("we will pansharpen the PAN images beware")
        if sceneparams.msi_to_pan.name != "identity":
            print(
                "we used the msi_to_pan transform",
                sceneparams.msi_to_pan.name,
                " but identity is the only one that should be working beware",
            )
        pansharp_method = load_pansharp(pansharp_cfg=opt.pansharp_cfg)
        if opt.flowmatching.apply_flowmatching:
            print("we will do flowmatching and pansharpening, will it be working?")

    if opt.flowmatching.apply_flowmatching:
        warper = performOpticalmatching(
            perform_cst_displacement=opt.flowmatching.perform_cst_displacement,
            mode=opt.flowmatching.mode,
            model_name=opt.flowmatching.model_name,
        )
        print("we will apply flowmatching")
    early_stopping_value = False
    for iteration in range(first_iter, opt.iterations + 1):
        # Every 1000 its we increase the levels of SH up to a maximum degree

        if (
            sceneparams.camera_params.learn_wv_transform
            and iteration == opt.iterstart_learn_wv_transform
        ):
            unfreeze_wv_transform(scene=scene, opt=opt)
            print("we unfreeze the WV transform")

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam: "MS_affine_cameras" = viewpoint_stack.pop(
            randint(0, len(viewpoint_stack) - 1)
        )

        # if the iteration is 5000,unfreeze the msi_to_pan parameters
        if iteration == opt.iterstart_learn_msitopan_params:
            print("WE UNFREEZE THE MSI TO PAN THING ")
            for cam in scene.getTrainCameras():
                pan_cam = cam.get_pan_cameras()

                pan_cam.unfreeze_msi_to_pan()

        # we tried to maintain both a PAN/MSI rendering, but it would add too much line and we add bigger MAE value.
        list_cam: "List[AffineCamera]" = get_list_cam(viewpoint_cam, opt)
        loss = 0
        for cam in list_cam:
            # print("cam name",cam.image_name)
            bg = torch.rand((5), device="cuda") if opt.random_background else background
            if opt.copy_background_firschan:
                for c in range(3):
                    bg[c] = bg[0]
            bg[3] = cam.altitude_bounds[0].item()
            bg[4] = 0.0
            render_pkg = render(cam, gaussians, pipe, bg)
            raw_render = render_pkg["render"][:3]
            altitude_render = render_pkg["render"][3]
            accumulated_opacity_render = render_pkg["render"][4]
            rendered_uva = torch.stack(cam.UV_grid + (altitude_render,), dim=-1)

            # ----- Compute loss -----
            # Initialize all extra losses to zero
            L_opacity = 0
            L_opacity_radii = 0
            L_sun_altitude_resample = 0
            L_sun_rgb_resample = 0
            L_new_altitude_resample = 0
            L_new_rgb_resample = 0
            L_TV_altitude = 0
            L_erank = 0
            L_nll = 0
            L_translucentshadows = 0
            sun_altitude_diff = None
            L_accumulated_opacity = 0
            L_pan = 0
            L_gradient_pan = 0
            L_pansharp = 0
            Lphotometric = 0
            L_flowmatch = 0
            # start compute the losses
            # Sun camera rendering and losses
            if iteration > opt.iterstart_shadowmapping:
                sun_camera, camera_to_sun = cam.get_sun_camera()
                sun_rgb_sample, sun_altitude_sample, sun_uv = (
                    render_resample_virtual_camera(
                        virtual_camera=sun_camera,
                        cam2virt=camera_to_sun,
                        rendered_uva=rendered_uva,
                        gaussians=gaussians,
                        pipe=pipe,
                        background=bg,
                    )
                )

                sun_altitude_diff = altitude_render - sun_altitude_sample
                if iteration > opt.iterstart_L_sun_resample:  # not used yet
                    L_sun_altitude_resample, L_sun_rgb_resample = suncamera_l(
                        raw_render=raw_render,
                        sun_rgb_sample=sun_rgb_sample,
                        sun_altitude_diff=sun_altitude_diff,
                        sun_uv=sun_uv,
                    )
            # * render here
            output = cam.render_pipeline(
                raw_render=raw_render,
                sun_altitude_diff=sun_altitude_diff,
            )
            opacity = gaussians.get_opacity.squeeze()
            image = output["final"]
            # image = output["shaded"]
            gt_image = cam.original_image.cuda()

            if sceneparams.repeat_gt and cam.image_type == "pan":
                gt_image = gt_image.repeat(3, 1, 1)
            if cam.image_type == "pan" and opt.apply_pansharp and opt.load_pan:
                if not cam.postfix_original_image:
                    gt_image = pansharp_method(
                        img_pan=gt_image.clone(),
                        img_msi=viewpoint_cam.get_msi_cameras().original_image.clone(),
                    )
                    cam.postfix_original_image = True
                    cam.original_image = gt_image
            #! Here we propose as a postfix to perform flow matching
            if (
                opt.flowmatching.apply_flowmatching
                and iteration > opt.iterstart_flowmatching
                and iteration < opt.flowmatching.iterend_flowmatching
            ):
                if (cam.image_type == "msi" and opt.flowmatching.flowmatch_msi) or (
                    cam.image_type == "pan" and opt.flowmatching.flowmatch_pan
                ):
                    predicted_flows, gt_image, image = perform_flow_matching(
                        opt, warper, image, gt_image
                    )
                    if abs(
                        predicted_flows
                    ).mean() > opt.flowmatching.max_value_flow and (
                        iteration < 2000 or iteration % 1000 == 0
                    ):
                        print(
                            "for cam image",
                            cam.image_name,
                            cam.image_type,
                            " we have a huge predicted flows, it will perhaps crash?",
                            " we discard this image and continue",
                        )
            if iteration > opt.iterstart_L_accumulated_opacity:  # not used yet
                L_accumulated_opacity = accumulatedopacity_l(
                    accumulated_opacity_render=accumulated_opacity_render
                )
            # Random camera rendering and losses
            if iteration > opt.iterstart_L_new_resample:
                new_camera, camera_to_new = cam.sample_random_camera(
                    opt.virtual_camera_extent
                )
                L_new_altitude_resample, L_new_rgb_resample = randomcamera_l(
                    new_camera=new_camera,
                    true_cam=cam,
                    camera_to_new=camera_to_new,
                    rendered_uva=rendered_uva,
                    gaussians=gaussians,
                    pipe=pipe,
                    bg=bg,
                    altitude_render=altitude_render,
                    raw_render=raw_render,
                    gt_image=gt_image,
                    shaded=output["shaded"],
                )

            # TV computation
            if iteration > opt.iterstart_L_TV_altitude:  # not used
                L_TV_altitude = TV_l(
                    altitude_render=altitude_render,
                )
            # Opacity loss, Eq 20 ?
            if (
                iteration > opt.iterstart_L_opacity
                and iteration < opt.iterend_L_opacity
            ):
                L_opacity = opacity_l(gaussians=gaussians)
            if (
                iteration > opt.iterstart_L_opacity_radii
                and iteration < opt.iterend_L_opacity_radii
            ):
                radii = render_pkg["radii"]
                L_opacity_radii = opacity_radii_l(gaussians=gaussians, radii=radii)
            if iteration > opt.iterstart_L_erank:  # not used yet
                L_erank = erank_l(gaussians=gaussians)
            if (
                iteration > opt.iterstart_L_flowmatch
                and iteration > opt.iterstart_flowmatching
                and iteration < opt.iterend_L_flowmatch
            ):
                L_flowmatch = flowmatch_l(flow=predicted_flows)
            if output["shadowmap"] is not None:
                shadowmap = output["shadowmap"]
                L_translucentshadows = translucentshadows_l(shadowmap=shadowmap)
            # Loss

            Ll1 = l1_loss(image, gt_image)
            metric_dict["L1"] += Ll1.detach().item()

            if opt.iterstart_L_photometric:
                Lphotometric = photometric_l(gt_image=gt_image, image=image, Ll1=Ll1)
                metric_dict["photometric"] += Lphotometric.detach().item()
                metric_dict["n_photo"] += 1
            L_altitude_reference = (
                (altitude_render - cam.reference_altitude).abs().mean()
            )
            if iteration > opt.iterstart_L_nll:
                betaprime = (cam.transient_mask.clip(0, 1) + 1e-3).square()  #  + 0.05

                L_nll = torch.nn.functional.gaussian_nll_loss(
                    input=image,
                    target=gt_image,
                    var=betaprime.unsqueeze(0).repeat(image.shape[0], 1, 1),
                )
                if iteration % 101 == 0 and cam.use_transient:
                    print(
                        "the average transient_mask is ",
                        cam.transient_mask.mean().item(),
                        "for cam.img_name",
                        cam.image_name,
                    )
                    print("his std is ", cam.transient_mask.std().item())
                    print("the loss is", opt.w_L_nll * L_nll)
            inter_loss = (
                opt.w_L_photometric * Lphotometric
                # + L_altitude_reference
                + opt.w_L_opacity * L_opacity
                + opt.w_L_opacity_radii * L_opacity_radii
                + opt.w_L_sun_altitude_resample * L_sun_altitude_resample
                + opt.w_L_sun_rgb_resample * L_sun_rgb_resample
                + opt.w_L_new_altitude_resample * L_new_altitude_resample
                + opt.w_L_new_rgb_resample * L_new_rgb_resample
                + opt.w_L_TV_altitude * L_TV_altitude  # not used
                + opt.w_L_erank * L_erank  # not used
                + opt.w_L_nll * L_nll  # not used
                + opt.w_L_translucentshadows * L_translucentshadows
                + opt.w_L_accumulated_opacity * L_accumulated_opacity
                + opt.w_L_flowmatch * L_flowmatch
            )
            # loss += inter_loss.mean()
            loss += inter_loss.mean().detach()  #!

            inter_loss.mean().backward()
            # compute the psnr here
            if cam.image_type == "pan":
                pan_psnr += psnr(image, gt_image).mean().float().item()
                n_pan += 1
                if pan_psnr < 0 and iteration > 200:
                    print("we got a negative PSNR for camera", cam.image_name)
                pan_ssim += ssim(image, gt_image).item()

                pan_img = image
                # gt_pan_img = gt_image
                # pan_raw_render = raw_render

            elif cam.image_type == "msi":
                n_msi += 1
                msi_psnr += psnr(image, gt_image).mean().double().item()
                msi_ssim += ssim(image, gt_image).item()
            else:
                raise ValueError(
                    f"Unknown camera type {cam.image_type}, should be either 'pan' or 'msi'"
                )
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Lphotometric_for_log = (
                0.4 * Lphotometric.item() + 0.6 * ema_Lphotometric_for_log
            )
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "Lphotometric": f"{ema_Lphotometric_for_log:.{7}f}",
                        "mae": f"{mae:.{7}f}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Logging
            if tb_writer and iteration % GS_loger.tb_log_interval == 0:
                # TODO: use log_loss

                metric_dict["photometric"] = metric_dict["photometric"] / max(
                    1, metric_dict["n_photo"]
                )
                metric_dict["L1"] = metric_dict["L1"] / max(1, metric_dict["n_photo"])
                metric_dict["pan_psnr"] = pan_psnr / max(1, n_pan)
                metric_dict["pan_ssim"] = pan_ssim / max(1, n_pan)
                metric_dict["msi_psnr"] = msi_psnr / max(1, n_msi)
                metric_dict["msi_ssim"] = msi_ssim / max(1, n_msi)
                tb_writer.add_scalar(
                    "number of gaussians", len(gaussians._xyz), iteration
                )
                tb_writer.add_scalar(
                    "Lphotometric",
                    metric_dict["photometric"],
                    iteration,
                )
                tb_writer.add_scalar("loss/LNLL", L_nll, iteration)
                tb_writer.add_scalar("Ll1", metric_dict["L1"], iteration)
                tb_writer.add_scalar("TotalLoss", loss, iteration)
                tb_writer.add_scalar(
                    "ref/L_altitude_reference", L_altitude_reference, iteration
                )
                tb_writer.add_scalar("meanopacity", opacity.mean().item(), iteration)
                tb_writer.add_scalar("loss/L_opacity", L_opacity, iteration)

                tb_writer.add_scalar(
                    "loss/L_sun_altitude_resample", L_sun_altitude_resample, iteration
                )  #!
                tb_writer.add_scalar(
                    "loss/L_sun_rgb_resample", L_sun_rgb_resample, iteration
                )
                tb_writer.add_scalar(
                    "loss/L_translucentshadows", L_translucentshadows, iteration
                )
                tb_writer.add_scalar(
                    "loss/L_new_altitude_resample", L_new_altitude_resample, iteration
                )
                tb_writer.add_scalar(
                    "loss/L_new_rgb_resample", L_new_rgb_resample, iteration
                )  #!

                tb_writer.add_scalar("loss/L_pan", L_pan, iteration)
                tb_writer.add_scalar("loss/L_gradient_pan", L_gradient_pan, iteration)

                tb_writer.add_scalar(
                    "train_metrics/msi_psnr", metric_dict["msi_psnr"], iteration
                )
                tb_writer.add_scalar(
                    "train_metrics/msi_ssim", metric_dict["msi_ssim"], iteration
                )

                tb_writer.add_scalar(
                    "train_metrics/pan_psnr", metric_dict["pan_psnr"], iteration
                )
                tb_writer.add_scalar(
                    "train_metrics/pan_ssim", metric_dict["pan_ssim"], iteration
                )

                #! Beware where yo ugonna put it :
                # Early stopping
                if opt.early_stopping.use_early_stopping:
                    early_stopping_value = early_stopper(metric_dict=metric_dict)
                    if early_stopping_value:  # we stop
                        opt.iterations = iteration  # to be able to save the model
                        saving_iterations = list(
                            set(saving_iterations) | set([iteration])
                        )

                # we reset here the metric dict
                pan_psnr = 0
                pan_ssim = 0
                n_pan = 0
                msi_psnr = 0
                msi_ssim = 0
                n_msi = 0
                metric_dict = {
                    "mae": mae,
                    "mae_wtree": mae_wtree,
                    "pan_ssim": pan_ssim,
                    "pan_psnr": pan_psnr,
                    "msi_psnr": msi_psnr,
                    "msi_ssim": msi_ssim,
                    "photometric": 0.0,
                    "L1": 0.0,
                    "n_photo": 0,
                }
            if iteration in GS_loger.big_testing_iterations:
                with torch.no_grad():
                    training_report(
                        tb_writer=tb_writer,
                        l1_loss=l1_loss,
                        testing_iterations=GS_loger.big_testing_iterations,
                        scene=scene,
                        pipeline=pipe,
                        iteration=iteration,
                        opt=opt,
                        background=background,
                        gaussians=gaussians,
                        sceneparams=sceneparams,
                        mae_computer=mae_computer,
                    )  #!

            # Color normalization before saving for the last time
            if iteration == opt.iterations:
                if opt.normalize_colors_before_saving:
                    normalize_before_saving(scene=scene, gaussians=gaussians)
                    print("done global cc alignment")
                else:
                    print("not normalizing colors before saving")
            # Saving
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                # in train.py we saved the color correction parameters, now we saved the whole camera parameters.
                # perform the final train_to_test cc converter if needed
                scene.train_to_test_cc_converter.perform_cc_to_test(
                    train_viewpoints=scene.getTrainCameras(),
                    test_cameras=scene.getTestCameras(),
                    opt=opt,
                )
                # save global camera parameters :
                camera_params_path = os.path.join(
                    scene.model_path, "camera_params/iteration_{}".format(iteration)
                )
                os.makedirs(camera_params_path, exist_ok=True)
                torch.save(
                    [
                        {
                            "image_name": c.image_name,
                            "state_dict": c,
                        }
                        for c in scene.getTrainCameras() + scene.getTestCameras()
                    ],
                    os.path.join(camera_params_path, "camera_params.pth"),
                )

                # Save the optimizer state
                optimizer_path = os.path.join(
                    scene.model_path, "optimizer/iteration_{}".format(iteration)
                )
                os.makedirs(optimizer_path, exist_ok=True)
                torch.save(
                    {
                        "gaussians": gaussians.optimizer.state_dict(),
                        "color_correction": camera_optimizer.state_dict(),
                        # "msi_pan": msi_pan_optimizer.state_dict(),
                    },
                    os.path.join(optimizer_path, "optimizer.pth"),
                )

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                camera_optimizer.step()
                # msi_pan_scheduler.step()

                gaussians.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)
                camera_optimizer.zero_grad(set_to_none=True)

            # Pruning
            if iteration < opt.densify_until_iter:
                if opt.only_prune:
                    transparent_mask = gaussians._opacity.squeeze() < opt.min_opacity
                    if transparent_mask.any():
                        gaussians.prune_points(transparent_mask)
                else:
                    radii = render_pkg["radii"]

                    visibility_filter = render_pkg["visibility_filter"]
                    viewspace_point_tensor = render_pkg["viewspace_points"]
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )

                    gaussians.add_densification_stats(
                        viewspace_point_tensor, visibility_filter
                    )
                    # we put back densify_and_split
                    if (
                        iteration > opt.densification_strategy.densify_from_iter
                        and iteration
                        % opt.densification_strategy.densification_interval
                        == 0
                    ):
                        size_threshold = (
                            20 if iteration > opt.opacity_reset_interval else None
                        )
                        gaussians.densify_and_prune(
                            grad_threshold=opt.densification_strategy.densify_grad_threshold,
                            min_opacity=0.005,  # opt.min_opacity,
                            screen_size_threshold=scene.cameras_extent,
                            max_screen_size=size_threshold,
                            radii=radii,
                            scene_extent=scene.cameras_extent,
                        )
                    transparent_mask = gaussians._opacity.squeeze() < opt.min_opacity
                    if transparent_mask.any():
                        gaussians.prune_points(transparent_mask)
            # Apply the flowmatching shift to the gaussians
            if iteration == opt.itr_apply_flowmatching_to_affine:
                print("we adjust the affine using the flowmatching parameters")
                print("HELLO")
                scene = adjust_affine_from_flow(
                    scene=scene,
                    warper=warper,
                    gaussians=gaussians,
                    pipe=pipe,
                    opt=opt,
                    bg=bg,
                )

            # added back the opacity reset_interval
            if (
                opt.opacity_reset_interval >= 0
                and iteration % opt.opacity_reset_interval == 0
                and iteration < opt.iterend_opacity_reset_interval
            ):
                print("WE RESET OPACITY WHAT GONNA HAPPEN?")
                gaussians.reset_opacity()
            if iteration == opt.color_reset_iterations:
                # do a fancy color reset
                color_reset(scene=scene, gaussians=gaussians, pipe=pipe)
                print("done color reset")

            if iteration in testing_iterations:
                test_background = torch.tensor(
                    [
                        1.0,
                        0.0,
                        1.0,
                        scene.get_reference_camera()
                        .get_pan_cameras()
                        .altitude_bounds[0]
                        .item(),
                        0,
                    ],
                    dtype=torch.float32,
                    device="cuda",
                )
                nadir_cam = [
                    c.get_pan_cameras()
                    for c in scene.getTestCameras()
                    if "Nadir" in c.image_name
                ]
                out = render_all_views(nadir_cam, gaussians, pipe, bg=test_background)
                nadir_cam = nadir_cam[0]
                altitude_render = out[0]["altitude_render"]  # (H,W)

                os.makedirs(
                    os.path.join(scene.model_path, "altitude_records"), exist_ok=True
                )
                nadir_rendered_uva = torch.stack(
                    nadir_cam.UV_grid + (altitude_render,), dim=-1
                )
                scene_params = [
                    scene.scene_shift,
                    scene.scene_scale,
                    scene.scene_n,
                    scene.scene_l,
                ]
                profile, dsm = compute_dsm_from_view(
                    view=nadir_cam,
                    rendered_uva=nadir_rendered_uva,
                    scene_name=sceneparams.scene_name,
                    scene_params=scene_params,
                )
                log_dsm_path = os.path.join(
                    scene.model_path,
                    "altitude_records",
                    f"altitude_render_{iteration:02}.tif",
                )
                with rasterio.open(log_dsm_path, "w", **profile) as f:
                    f.write(dsm[:, :, 0], 1)
                mae, _, _, profile, pred_dsm = mae_computer.compute_mae_from_path(
                    pred_dsm_path=log_dsm_path
                )
                mae_wtree, _, _, _ = mae_computer.compute_mae_from_pred_dsm(
                    pred_dsm=pred_dsm, profile=profile, force_use_tree_mask=True
                )  # TODO: remove it when releasing the code

                mae_computer.remove_dsm(pred_dsm_path=log_dsm_path)

                tb_writer.add_scalar("MAE", mae, iteration)
                tb_writer.add_scalar("MAE_notree", mae_wtree, iteration)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                print(
                    "path saved:",
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )
            if opt.early_stopping.use_early_stopping and early_stopping_value:
                print(f"we stopped at iteration {iteration} because of early stopping")
                break


def prepare_output_and_logger(sceneparams):
    if not sceneparams.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        sceneparams.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(sceneparams.model_path))
    os.makedirs(sceneparams.model_path, exist_ok=True)
    # instantiate the consive recursively
    nv_vars = recursive_args_from_pg(sceneparams)
    with open(os.path.join(sceneparams.model_path, "cfg_args.yaml"), "w") as cfg_log_f:
        OmegaConf.save(config=nv_vars, f=cfg_log_f.name)
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(sceneparams.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    l1_loss,
    testing_iterations,
    scene: "Scene",
    opt,
    background,
    gaussians,
    pipeline,
    sceneparams,
    mae_computer,
):
    # Report test and samples of training set
    torch.cuda.empty_cache()
    validation_configs = (
        {
            "name": "test",
            "cameras": [
                cam for cam in scene.getTestCameras() if "Nadir" not in cam.image_name
            ],
        },
        {
            "name": "train",
            "cameras": scene.getTrainCameras(),
            #     [
            #     # scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
            #     # for idx in range(4, 12, 3)
            # ],
        },
    )
    # Update the test color correction if needed
    scene.train_to_test_cc_converter.perform_cc_to_test(
        train_viewpoints=scene.getTrainCameras(),
        test_cameras=scene.getTestCameras(),
        opt=opt,
    )
    print("converted the train to test color correction before testing")
    for config in validation_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            l1_test = {"pan": 0.0, "msi": 0.0}
            psnr_test = {"pan": 0.0, "msi": 0.0}
            for idx, viewpoint in enumerate(config["cameras"]):
                # first render the image
                for cam in get_list_cam(viewpoint, opt):
                    bg = (
                        torch.rand((5), device="cuda")
                        if opt.random_background
                        else background
                    )

                    bg[3] = cam.altitude_bounds[0].item()
                    bg[4] = 0.0
                    render_pkg = render(cam, gaussians, pipeline, bg)
                    raw_render = render_pkg["render"][:3]
                    altitude_render = render_pkg["render"][3]

                    rendered_uva = torch.stack(cam.UV_grid + (altitude_render,), dim=-1)
                    sun_camera, cam2sun = cam.get_sun_camera()

                    sun_rgb_sample, sun_altitude_sample, _, sunpov = (
                        render_resample_virtual_camera(
                            virtual_camera=sun_camera,
                            cam2virt=cam2sun,
                            rendered_uva=rendered_uva,
                            gaussians=gaussians,
                            pipe=pipeline,
                            background=bg,
                            return_extra=True,
                        )
                    )
                    sun_altitude_diff = altitude_render - sun_altitude_sample

                    renderings = cam.render_pipeline(
                        raw_render=raw_render,
                        sun_altitude_diff=sun_altitude_diff,
                    )
                    image = renderings["shaded"]

                    gt_image = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_v_{}_{}/render".format(
                                viewpoint.image_name[
                                    : min(len(viewpoint.image_name), 5)
                                ],
                                cam.image_type,
                            ),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_v_{}_{}/ground_truth".format(
                                    viewpoint.image_name[
                                        : min(len(viewpoint.image_name), 5)
                                    ],
                                    cam.image_type,
                                ),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test[cam.image_type] += l1_loss(image, gt_image).mean().double()
                    psnr_test[cam.image_type] += psnr(image, gt_image).mean().double()
            for key in psnr_test:
                psnr_test[key] /= len(config["cameras"])
                l1_test[key] /= len(config["cameras"])
            print(
                "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                    iteration, config["name"], l1_test, psnr_test
                )
            )
            if tb_writer:
                for key in psnr_test:
                    tb_writer.add_scalar(
                        config["name"] + f"/loss_viewpoint - l1_loss_{key}",
                        l1_test[key],
                        iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + f"/loss_viewpoint - psnr_{key}",
                        psnr_test[key],
                        iteration,
                    )
        else:
            print("no cameras for this validation config ", config["name"])
    # here we log the absolute rdsm w.r.t to the gt
    test_background = torch.tensor(
        [
            1.0,
            0.0,
            1.0,
            scene.get_reference_camera().get_pan_cameras().altitude_bounds[0].item(),
            0,
        ],
        dtype=torch.float32,
        device="cuda",
    )
    nadir_cam = [
        c.get_pan_cameras() for c in scene.getTestCameras() if "Nadir" in c.image_name
    ]
    out = render_all_views(nadir_cam, gaussians, pipe=pipeline, bg=test_background)
    nadir_cam = nadir_cam[0]
    altitude_render = out[0]["altitude_render"]  # (H,W)

    nadir_rendered_uva = torch.stack(nadir_cam.UV_grid + (altitude_render,), dim=-1)
    scene_params = [
        scene.scene_shift,
        scene.scene_scale,
        scene.scene_n,
        scene.scene_l,
    ]
    profile, dsm = compute_dsm_from_view(
        view=nadir_cam,
        rendered_uva=nadir_rendered_uva,
        scene_name=sceneparams.scene_name,
        scene_params=scene_params,
    )
    log_dsm_path = os.path.join(
        scene.model_path,
        "altitude_records",
        f"altitude_render_{iteration:02}.tif",
    )
    os.makedirs(os.path.dirname(log_dsm_path), exist_ok=True)
    with rasterio.open(log_dsm_path, "w", **profile) as f:
        f.write(dsm[:, :, 0], 1)
    mae, diff, rdsm, profile, _ = mae_computer.compute_mae_from_path(
        pred_dsm_path=log_dsm_path
    )

    mae_computer.remove_dsm(pred_dsm_path=log_dsm_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axes[0].imshow(rdsm, cmap="viridis")
    axes[0].set_title("RDSM")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot abs(diff)
    im1 = axes[1].imshow(abs(diff), cmap="viridis", vmax=5)
    axes[1].set_title(f"Abs(diff), MAE={mae:.3f} m")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    tb_writer.add_figure("RDSM_and_diff", fig, global_step=iteration)
    plt.close(fig)

    torch.cuda.empty_cache()
    return


@hydra.main(version_base="1.2", config_path="gs_config", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    model = ModelParams(cfg.get("model", None))
    opt = OptimizationParams(cfg.get("optimization", None))
    cmlparams = ClearmlParams(cfg.get("clearml", None))
    pipeline: "PipelineParams" = PipelineParams(cfg=cfg.get("pipeline", None))
    test_iterations_default = (
        list(range(0, 100)) + list(range(100, 1000, 10)) + list(range(1000, 10000, 50))
    )
    GS_loger: "loggingGS" = cfg.get("gs_logger", None)
    test_iterations_default = (
        list(range(0, 100, 10)) + list(range(0, 100000, 100)) + [opt.iterations - 1]
    )
    print("the base path is ", cfg.path.basepath)
    test_iterations_default = sorted(list(set(test_iterations_default)))
    if CLEARML_FOUND and not pipeline.debug:
        from utils.clearml_utils import safe_init_clearml, connect_whole

        assert (
            cmlparams.task_name != ""
        ), "Please provide a task name for ClearML,got {}".format(cmlparams.task_name)
        task = safe_init_clearml(
            project_name=cmlparams.project_name,
            task_name=cmlparams.task_name,
            tags=cmlparams.tags,
        )
        connect_whole(
            cfg=cfg,
            task=task,
            name_hyperparams_summary="train config",
            name_connect_cfg="whole train cfg",
        )
        # task.connect(cfg,name="test_train")
    else:
        print(
            " you probably are in debug mode"
        )
    print("Optimizing " + cfg.model.model_path)
    if not cfg.run_train:
        print("run_train is set to False, exiting...")
        return
    # Initialize system state (RNG)
    safe_state(cfg.quiet, seed=cfg.seed)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)
    training(
        sceneparams=model,
        opt=opt,
        pipe=pipeline,
        GS_loger=GS_loger,
        testing_iterations=test_iterations_default,
        saving_iterations=cfg.save_iterations,
        checkpoint_iterations=cfg.checkpoint_iterations,
        start_checkpoint=cfg.start_checkpoint,
        debug_from=cfg.debug_from,
    )
    # All done
    print("\nTraining complete.")
    if CLEARML_FOUND and not pipeline.debug:
        print("Attempting to close clearml task")
        # print("task url",task.get_web_a)

        task.close()
        print("ClearML task closed")


if __name__ == "__main__":
    main()
