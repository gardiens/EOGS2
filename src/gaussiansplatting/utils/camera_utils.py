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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from scene.dataset_readers.dataset_affine import AffineCameraInfo, MSAffineCameraInfo
import torch

WARNED = False


def get_list_cam(viewpoint_cam, opt) -> "List[AffineCamera]":
    list_cam = []

    if opt.load_msi:
        list_cam.append(viewpoint_cam.get_msi_cameras())

    if opt.load_pan:
        list_cam.append(viewpoint_cam.get_pan_cameras())

    return list_cam


def loadCam(args, id, cam_info, resolution_scale):
    if isinstance(cam_info, AffineCameraInfo):
        assert args.resolution in [
            1,
            -1,
        ], f" resolution should be 1 or -1, got {args.resolution}"
        assert (
            resolution_scale == 1
        ), f" AffineCameraInfo should not be rescaled, got {resolution_scale}"
        #! WHY THE LOADING DOESNT CRASH ON GRAY SCALE?
        affine_camera = cam_info.convert_to_affine_camera(
            image=torch.from_numpy(cam_info.image).permute(2, 0, 1).float(),
            gt_alpha_mask=None,
            data_device=args.data_device,
        )
        return affine_camera
    # if isinstance(cam_info, MSAffineCameraInfo):
    #     assert args.resolution in [
    #         1,
    #         -1,
    #     ], f" resolution should be 1 or -1, got {args.resolution}"
    #     assert (
    #         resolution_scale == 1
    #     ), f" MSAffineCameraInfo should not be rescaled, got {resolution_scale}"
    #     #! WHY THE LOADING DOESNT CRASH ON GRAY SCALE?
    #     affine_camera = cam_info.convert_to_MS_affine_camera(
    #         images=cam_info.images,
    #         data_device=args.data_device,
    #     )
    #     return affine_camera
    print("FOR EOGS, it shouldn't happen here, type of camera", type(cam_info))
    print("Debbug camera affine info", type(cam_info) == AffineCameraInfo)
    print("debug camera msi 2 ", MSAffineCameraInfo == type(cam_info))
    print("debug 3", isinstance(cam_info, MSAffineCameraInfo))
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = (
            round(orig_w / (resolution_scale * args.resolution)),
            round(orig_h / (resolution_scale * args.resolution)),
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
