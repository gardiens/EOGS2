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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos

from typing import List, TYPE_CHECKING
from scene.cameras import AffineCamera
from scene.dataset_readers.dataset_readers import readAffineSceneInfo

from scene.dataset_readers.dataset_MS_affine import readMSAffineSceneInfo

if TYPE_CHECKING:
    from scene.dataset_readers.dataset_readers import SceneInfo, MSSceneInfo


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration: None = None,
        shuffle: "bool" = False,
        resolution_scales: "List[float]" = [1.0],
    ) -> None:
        """B :param path: Path to colmap scene main folder."""
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # * read scene information
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )
        elif os.path.exists(
            os.path.join(args.source_path, "affine_models.json")
        ):  # * Affine camera models
            print("Found affine_models.json file, assuming Affine data set!")
            load_image_type = {
                "pan": args.load_pan,
                "msi": args.load_msi,
            }
            if args.load_pan:
                images_path = {
                    "msi": args.images_msi_path,
                    "pan": args.images_pan_path,
                }
                scene_info: "MSSceneInfo" = readMSAffineSceneInfo(
                    path=args.source_path,
                    images_path=images_path,
                    eval=args.eval,
                    camera_params=args.camera_params,
                    load_image_type=load_image_type,
                )
            else:  # fall back to usual affine scenario
                scene_info: "SceneInfo" = readAffineSceneInfo(
                    path=args.source_path,
                    images_msi_path=args.images_msi_path,
                    images_pan_path=args.images_pan_path,
                    eval=args.eval,
                    camera_params=args.camera_params,
                    load_image_type=load_image_type,
                )

            # the called function is readAffineSceneInfo
        else:
            assert False, f"Could not recognize scene type! got {args.source_path}"

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file:
                with open(
                    os.path.join(self.model_path, "input.ply"), "wb"
                ) as dest_file:
                    dest_file.write(src_file.read())

        if shuffle:
            random.shuffle(
                scene_info.train_cameras
            )  # Multi-res consistent random shuffling
            random.shuffle(
                scene_info.test_cameras
            )  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.scene_scale = scene_info.nerf_normalization["scale"]
        self.scene_shift = scene_info.nerf_normalization["center"]
        self.scene_n = scene_info.nerf_normalization["n"]
        self.scene_l = scene_info.nerf_normalization["l"]

        # * actually load the cameras !
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )
        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        else:
            self.gaussians.create_from_pcd(
                pcd=scene_info.point_cloud,
                cam_infos=scene_info.train_cameras,
                spatial_lr_scale=self.cameras_extent,
                opacity_init_value=args.opacity_init_value,
            )

    def save(self, iteration: int) -> None:
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale: float = 1.0) -> "List[AffineCamera]":
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0) -> "List[AffineCamera]":
        return self.test_cameras[scale]
