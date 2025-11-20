import os
import random
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams

from typing import List, TYPE_CHECKING
from scene.dataset_readers.dataset_readers import readAffineSceneInfo

from scene.dataset_readers.dataset_MS_affine import readMSAffineSceneInfo

if TYPE_CHECKING:
    from scene.dataset_readers.dataset_readers import SceneInfo
    from scene.dataset_readers.dataset_MS_affine import MSSceneInfo
    from utils.typing_utils import *

from scene import Scene
from scene.cameras.MS_affine_cameras import MSAffineCamera
from utils.rescaler.rescaler import load_rescaler, perform_rescaling
from utils.convert_color_correction import load_convert_color_correction_train_to_test


class MSScene(Scene):
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
        self.train_to_test_cc_converter: "perform_average_cc_test" = (
            load_convert_color_correction_train_to_test(
                name=args.train_to_test_cc_converter
            )
        )
        print("we use the train_to_test_cc_converter ", args.train_to_test_cc_converter)
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
        assert os.path.exists(
            args.source_path
        ), f"Source path {args.source_path} does not exist!"
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
            print("the source path is", args.source_path)
            if args.load_pan:
                print("we have MSI and PAN images, loading MS scene info")
                images_path = {
                    "msi": args.images_msi_path,
                    "pan": args.images_pan_path,
                }
                print("the images path is", images_path)
                print(
                    "need rescale",
                    args.need_rescale,
                    "should be false iif SYNEW in path ",
                )
                scene_info: "MSSceneInfo" = readMSAffineSceneInfo(
                    path=args.source_path,
                    images_path=images_path,
                    eval=args.eval,
                    camera_params=args.camera_params,
                    load_image_type=load_image_type,
                    need_rescale=args.need_rescale,
                    target_density=args.target_density,
                    scale_factor_z=args.scale_factor_z,
                    modelparams=args,
                )
            else:  # fall back to usual affine scenario
                scene_info: "SceneInfo" = readAffineSceneInfo(
                    path=args.source_path,
                    images_msi_path=args.images_msi_path,
                    images_pan_path=args.images_pan_path,
                    eval=args.eval,
                    camera_params=args.camera_params,
                    load_image_type=load_image_type,
                    need_rescale=args.need_rescale,
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
            # get the train and test cameras information

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
            self.train_cameras[resolution_scale] = self.cameraList_from_camInfos_MS(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = self.cameraList_from_camInfos_MS(
                scene_info.test_cameras, resolution_scale, args
            )
        # * load or create the gaussian model.
        if self.loaded_iter:
            # check that self.model_path is correct
            assert os.path.exists(
                self.model_path
            ), f"Model path {self.model_path} does not exist!"
            assert os
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
        # get the reference camera
        self.reference_camera = None
        self.set_reference_camera(
            train_cameras=self.train_cameras[resolution_scales[0]]
        )

        # TODO: perform the rescaling setup here
        print("we will perform rescaler with name ", args.rescaler_name)
        self.rescaler = load_rescaler(args.rescaler_name)
        train_cameras = self.getTrainCameras()
        perform_rescaling(
            train_cameras=train_cameras, rescaler=self.rescaler, args=args
        )

        # verbosefor affine cameras
        if args.share_color_correction:
            print("we will share the color correction over cameras")
        if args.share_worldview_transform:
            print("we will share the world view transform over cameras")

    def get_reference_camera(self) -> "MSAffineCamera":
        return self.reference_camera

    def set_reference_camera(self, train_cameras) -> None:
        for cam in train_cameras:
            if cam.is_reference_camera:
                self.reference_camera = cam
                return self.reference_camera
        raise ValueError(
            "No reference camera found in the training cameras. Please set one manually"
        )
        return self.reference_camera

    def save(self, iteration: int) -> None:
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale: float = 1.0) -> "List[MSAffineCamera]":
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0) -> "List[MSAffineCamera]":
        return self.test_cameras[scale]

    def cameraList_from_camInfos_MS(
        self, cam_infos, resolution_scale, args: "ModelParams"
    ) -> "List[MSAffineCamera]":
        camera_list = []
        keys = cam_infos[0].images.keys()
        for cam_info in cam_infos:
            camera_result = {}

            for k in keys:
                c = cam_info.images[k]
                camera_result[k] = MS_loadCam(args, id, c, resolution_scale, im_type=k)
            camera_result = MSAffineCamera(
                Affinecamera_dict=camera_result,
                is_reference_camera=cam_info.is_reference_camera,
                image_name=cam_info.image_name,
                data_device=args.data_device,
                args=args,
            )
            camera_list.append(camera_result)
            # loadCam(args,id,cam_info,resolution_scale=resolution_scale)

        return camera_list


from scene.dataset_readers.dataset_affine import AffineCameraInfo
import torch


def MS_loadCam(args, id, cam_info, resolution_scale, im_type="msi"):
    if type(cam_info) == AffineCameraInfo:
        assert args.resolution in [
            1,
            -1,
        ], f" resolution should be 1 or -1, got {args.resolution}"
        assert (
            resolution_scale == 1
        ), f" AffineCameraInfo should not be rescaled, got {resolution_scale}"

        if im_type == "msi":
            affine_camera = cam_info.convert_to_affine_camera(
                image=torch.from_numpy(cam_info.image).permute(2, 0, 1).float(),
                gt_alpha_mask=None,
                data_device=args.data_device,
                args=args,
            )
        elif im_type == "pan":
            affine_camera = cam_info.convert_to_pan_affine_camera(
                image=torch.from_numpy(cam_info.image).permute(2, 0, 1).float(),
                gt_alpha_mask=None,
                data_device=args.data_device,
                args=args,
            )
        return affine_camera
