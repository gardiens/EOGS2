import os
import numpy as np
import json
import iio
from scene.dataset_readers.dataset_utils import storePly, fetchPly, SceneInfo
from scene.cameras import AffineCamera
from scene.cameras.PAN_affine_cameras import PANAffineCamera
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_utils import *


@dataclass
class AffineCameraInfo:
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    is_reference_camera: bool
    reference_altitude: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    centerofscene_ECEF: np.array
    affine_coef: np.array
    affine_inter: np.array
    altitude_bounds: np.array
    min_world: np.array
    max_world: np.array
    sun_affine_coef: np.array
    sun_affine_inter: np.array
    camera_to_sun: np.array
    learn_wv_transform: bool
    use_only_rot: bool
    use_cc: bool
    load_sun: bool
    use_shadow: bool
    virtual_camera: bool

    def __repr__(self):
        # show everything except R, T, FovY, FovX
        return (
            f"AffineCameraInfo(uid={self.uid}, "
            f"image_name={self.image_name}, "
            f"width={self.width}, height={self.height}, "
            f"centerofscene_ECEF={self.centerofscene_ECEF}, "
            f"altitude_bounds={self.altitude_bounds}, "
            f"min_world={self.min_world}, "
            f"max_world={self.max_world}, "
            f"sun_affine_coef={self.sun_affine_coef}, "
            f"sun_affine_inter={self.sun_affine_inter}, "
            f"learn_wv_transform={self.learn_wv_transform}, "
            f"use_cc={self.use_cc})"
        )

    def __getattribute__(self, name):
        blacklist = ["R", "T", "FovY", "FovX"]
        if name in blacklist:
            raise AttributeError(f"Hai provato af accedere a {name}")
        return super().__getattribute__(name)

    def convert_to_affine_camera(
        self,
        image,
        gt_alpha_mask,
        data_device,
        args,
    ):
        "actual function that will convert the AffineCameraInfo to an AffineCamera object"

        return AffineCamera(
            self,
            image,
            gt_alpha_mask,
            data_device,
            is_reference_camera=self.is_reference_camera,
            args=args,  # pass the args to the AffineCamera
        )

    def convert_to_pan_affine_camera(
        self, image, gt_alpha_mask, data_device, args=None
    ):
        return PANAffineCamera(
            self,
            image,
            gt_alpha_mask,
            data_device,
            args=args,
        )


@dataclass
class MSAffineCameraInfo:
    images: "dict"  # pan : ,msi: List[str]
    is_reference_camera: bool
    image_name: str = None
    uid: int = None

    # def convert_to_MS_affine_camera(self,images,data_device):
    #     return MS_affine_cameras(
    #         self,
    #         Affinecamera_dict=images,
    #         is_reference_camera=self.is_reference_camera,
    #         image_name=self.image_name,
    #         data_device=data_device,

    #     )


_altitude_warning_issued = False  # Global or module-level variable


def load_altitude(metadata, img_path, img):
    global _altitude_warning_issued

    if "-NEW-" in img_path:
        reference_altitude = img_path.replace("-NEW-", "-SYNEW-")
    else:
        # assert '-SYNEW-' in img_path, "Reference altitude not found"
        reference_altitude = img_path
    reference_altitude = os.path.join(reference_altitude, "altitude", metadata["img"])
    if not os.path.exists(reference_altitude):
        reference_altitude = img.copy()[..., 0] * 0.0
        if not _altitude_warning_issued:
            print(
                "Warning: Reference altitude not found, using zeros, logged only once"
            )
            _altitude_warning_issued = True
    else:
        reference_altitude = np.squeeze(iio.read(reference_altitude))
    return reference_altitude


import rasterio


def load_img(metadata, img_path, need_rescale: bool = False):
    if (not metadata.get("virtual_camera", False)) and metadata["img"] != "Nadir":
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Image {img_path} not found! Please check the path and the metadata."
            )
        # if "SYNEW" in img_path:
        #     img = iio.read(img_path)
        # else:
        #     img = iio.read(img_path) / 255.0
        # read the img
        # If the img_path is in a compressed scanlines, we open with rasterio instead
        try:
            img = (
                rasterio.open(img_path).read().transpose(1, 2, 0)
            )  #! changed it but dangeorus
        except:
            img = iio.read(img_path)

        if need_rescale:
            img = img / 255.0
        if img.max() > 1.5:
            print(
                "Warning: Images with path",
                img_path,
                "seems to be greatly higher than 1, is it normal?",
                "the maximum value is",
                img.max(),
            )
        return img

    else:
        # Nadir image is not loaded, it is a special case

        img = np.zeros((metadata["height"], metadata["width"], 1))
        return img
    return None  # should it happen?


def load_caminfo(
    metadata,
    images: str,
    camera_params: "CameraParams" = None,
    load_sun: bool = True,
    need_rescale: bool = False,
):
    # load img
    if not os.path.exists(images):
        raise FileNotFoundError(f"Images directory {images} does not exist!")
    img_path = os.path.join(images, metadata["img"])
    img = load_img(metadata=metadata, img_path=img_path, need_rescale=need_rescale)
    # images: path of iamges

    # load altitude
    reference_altitude = load_altitude(img_path=img_path, img=img, metadata=metadata)

    # get the affine model parameters
    lm_coef_ = np.array(metadata["model"]["coef_"])
    lm_intercept_ = np.array(metadata["model"]["intercept_"])
    altitude_bounds = np.array([metadata["min_alt"], metadata["max_alt"]])
    min_world = np.array(metadata["model"]["min_world"])
    max_world = np.array(metadata["model"]["max_world"])

    if load_sun:
        sun_lm_coef_ = np.array(metadata["sun_model"]["coef_"])
        sun_lm_intercept_ = np.array(metadata["sun_model"]["intercept_"])
        camera_to_sun = np.array(metadata["sun_model"]["camera_to_sun"])

    else:
        sun_lm_coef_ = None
        sun_lm_intercept_ = None
        camera_to_sun = None
    virtual_camera = metadata.get("virtual_camera", False)
    caminfo = AffineCameraInfo(
        uid=None,
        R=None,
        T=None,
        FovY=None,
        FovX=None,
        image=img,
        is_reference_camera=False,
        reference_altitude=reference_altitude,
        image_path=img_path,
        image_name=metadata["img"].replace(".tif", ""),
        width=metadata["width"],
        height=metadata["height"],
        centerofscene_ECEF=np.array(metadata["centerofscene_UTM"]),
        affine_coef=lm_coef_,
        affine_inter=lm_intercept_,
        altitude_bounds=altitude_bounds,
        min_world=min_world,
        max_world=max_world,
        sun_affine_coef=sun_lm_coef_,
        sun_affine_inter=sun_lm_intercept_,
        camera_to_sun=camera_to_sun,
        learn_wv_transform=camera_params.learn_wv_transform,
        use_only_rot=camera_params.use_only_rot,
        use_cc=camera_params.use_cc,
        load_sun=load_sun,
        use_shadow=camera_params.use_shadow,
        virtual_camera=virtual_camera,
    )
    return caminfo, min_world, max_world


def initalize_pcd(
    root_path, metadata, min_world, max_world, ply_path, target_density=0.13
):
    if True:  # not os.path.exists(ply_path):
        # In the to_affine.py script we normalized the scene so that:
        # - it is strictly inside [-1,1]^3
        # - it is euclidean (as ECEF) and it unit of measures are meters/metadata['model']['scale']
        # - the box is aligned with longitude and latitude
        # Moreover metadata['model']['min_world'] and metadata['model']['max_world']
        # contain the actual bounds of the scene in [-1,1]^3

        # We now generate random points inside the "inner" bounds of the scene keeping a 10% margin
        # It is IMPORTANT to have a truly uniform distribution inside the inner bbox,
        # meaning that the denisty should be isotropic and constant.
        # Otherwise the initialization of the scales won't work properly.

        # To do so, we start with a uniform distribution in [-1,1]^3 and
        # we keep only the points that are inside the inner bbox.
        # We aim for a target density (expressed in gaussians per true cubic meter).
        # As the distribution is uniform, the total points that should be generated is:
        # N = rho_target * V_out
        # This ensures that the density (both in the inner bbox and in the outer bbox) is correct.
        # There is a catch: we are working in normalized UTM coordinates by a scale factor.

        # target_density = 0.13  # 0.13 gaussians per true cubic meter.
        scale = metadata["model"]["scale"]
        sides = max_world * 1.1 - min_world * 1.1
        volume_inner = np.prod(sides)
        volume_outer = 2**3
        num_pts_to_be_generated = int(target_density * volume_outer * scale**3)
        xyz = np.random.rand(num_pts_to_be_generated, 3) * 2 - 1
        inside = np.all(xyz > min_world * 1.1, axis=1) & np.all(
            xyz < max_world * 1.1, axis=1
        )
        xyz = xyz[inside]
        # print("Number of points generated inside the inner bbox:", len(xyz))
        # print("Volume inner bbox:", volume_inner)
        # print("Density inside the inner bbox:", len(xyz) / (volume_inner * scale**3))
        print("Total density:", num_pts_to_be_generated / (volume_outer * scale**3))
        print("Expected density:", target_density)

        rgb = np.full((len(xyz), 3), 1.1)

        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None
    return pcd, xyz


def initalize_pcd_fromtoaffine(ply_path):
    # we load the pcd from the ply generated in to_affine.py
    pcd = fetchPly(ply_path)
    xyz = pcd.points
    return pcd, xyz


def split_traintest_caminfo(path, cam_infos, eval):
    if eval:
        with open(os.path.join(path, "train.txt"), "r") as trainsplit:
            trainsplit = trainsplit.read().splitlines()
            trainsplit = [x.replace(".json", "") for x in trainsplit]
        with open(os.path.join(path, "test.txt"), "r") as testsplit:
            testsplit = testsplit.read().splitlines()
            testsplit = [x.replace(".json", "") for x in testsplit]
        train_cam_infos = []
        test_cam_infos = []
        for caminfo in cam_infos[:-1]:
            if caminfo.image_name in trainsplit:
                train_cam_infos.append(caminfo)
            elif caminfo.image_name in testsplit:
                test_cam_infos.append(caminfo)
            else:
                raise RuntimeError("Image not in train or test split!")

        # add the last camera (perfectly nadir) to the test
        test_cam_infos.append(cam_infos[-1])
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    return train_cam_infos, test_cam_infos


def readAffineSceneInfo(
    path,
    images_msi_path,
    images_pan_path,
    eval,
    camera_params: "CameraParams",
    load_image_type: "dict",
    need_rescale: bool,
    target_density: float,
) -> "SceneInfo":
    images = images_msi_path
    with open(os.path.join(path, "affine_models.json"), "r") as metadatas:
        metadatas = json.load(metadatas)

    cam_infos = []
    for n, metadata in enumerate(metadatas):
        caminfo, min_world, max_world = load_caminfo(
            metadata=metadata,
            load_sun=True,
            images=images,
            camera_params=camera_params,
            need_rescale=need_rescale,
        )
        cam_infos.append(caminfo)

    # split in train test caminfo
    train_cam_infos, test_cam_infos = split_traintest_caminfo(
        path=path, cam_infos=cam_infos, eval=eval
    )

    # Setting the first train camera as the reference camera
    train_cam_infos[0].is_reference_camera = True
    # initialize the point clouds
    ply_path = os.path.join(path, "points3d.ply")
    pcd, xyz = initalize_pcd(
        root_path=path,
        metadata=metadata,
        max_world=max_world,
        min_world=min_world,
        ply_path=ply_path,
        target_density=target_density,
    )

    radius = np.linalg.norm(xyz - xyz.mean(axis=0), axis=1)
    radius = np.max(radius) * 2

    # The radius variable will be used for densification strategies but also for scaling the spatial_lr
    # 100 è troppo
    # 10 è ok
    # 1 è ok
    # radius = radius * 10

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization={
            "radius": radius,
            "scale": metadata["model"]["scale"],
            "center": metadata["model"]["center"],
            "n": metadata["model"]["n"],
            "l": metadata["model"]["l"],
        },
        ply_path=ply_path,
    )
    return scene_info
