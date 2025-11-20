import os
from glob import glob
import json
import numpy as np
import rpcm
from copy import deepcopy
import hydra
import iio
import rasterio

try:
    from .converter import PanConverter, storePly
except:
    from converter import PanConverter, storePly
from pathlib import Path


from omegaconf import OmegaConf

_has_logged_resolution_warning = True


from .utils_to_affine import (
    check_scale_factor,
    open_json_file,
    test,
    get_dir_vec_from_el_az,
    approximate_W2V_affine,
)
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def pipeline_msi(metadata, converter, model_W2V_coef_, model_W2V_intercept_):
    global _has_logged_resolution_warning

    img_name = metadata["img"]
    msi_image = iio.read(os.path.join(converter.msi_dir, img_name))

    # check the scale factor between the msi image and the metadata
    scale_factor_h = metadata["height"] / msi_image.shape[0]
    scale_factor_w = metadata["width"] / msi_image.shape[1]
    check_scale_factor(
        scale_factor_h=scale_factor_h,
        scale_factor_w=scale_factor_w,
        metadata=metadata,
        img=msi_image,
        img_name=img_name,
    )

    # change the resolution of the metadata if needed
    if converter.change_resolution:
        if (
            metadata["height"] != msi_image.shape[0]
            or metadata["width"] != msi_image.shape[1]
        ):
            if _has_logged_resolution_warning:
                print(
                    f"Changing resolution of the  msi metadata from {metadata['height']}x{metadata['width']} to {msi_image.shape[0]}x{msi_image.shape[1]}"
                )
        metadata["height"] = msi_image.shape[0]
        metadata["width"] = msi_image.shape[1]
    # as model_W2V mappes from  global coordinates to [-1,1], one could hope that we don't have to change the model  #TODO: check it
    # A=A/scale_factor_h
    # b=b/scale_factor_h

    # For the MSI image, we need to compute a affine approximation of the sun
    #########################################################
    ### Computing the affine approximation of the sun
    #########################################################
    # First we compute the change of basis M from lonlat to world coordinates
    # and the center of the scene in world coords
    # M, centerofscene_ECEF = converter.computeJacobian()

    M = np.eye(3)
    centerofscene_UTM = np.zeros(3)

    # Then we compute the sun direction in ECEF
    local_sun_direction = -get_dir_vec_from_el_az(
        elevation_deg=90 - float(metadata["sun_elevation"]),
        azimuth_deg=float(metadata["sun_azimuth"]),
    )
    sun_dir_ecef = M.T @ local_sun_direction
    sun_dir_ecef = sun_dir_ecef / (model_W2V_coef_ @ sun_dir_ecef)[2]

    # Then we compute the affine model for the sun
    Asun_dir_ecef = model_W2V_coef_ @ sun_dir_ecef
    myM = np.array([[1, 0, -Asun_dir_ecef[0]], [0, 1, -Asun_dir_ecef[1]], [0, 0, 1]])
    sun_A = myM @ model_W2V_coef_
    sun_b = (
        -sun_A @ centerofscene_UTM
        + model_W2V_coef_ @ centerofscene_UTM
        + model_W2V_intercept_
    )

    # Enrich the metadata with the new models
    metadata["virtual_camera"] = False
    metadata["centerofscene_UTM"] = centerofscene_UTM.tolist()
    metadata["model"] = {
        "coef_": model_W2V_coef_.tolist(),
        "intercept_": model_W2V_intercept_.tolist(),
        "scale": converter.scale,
        "n": converter.n,
        "l": converter.l,
        "center": converter.shift.tolist(),
        "min_world": converter.min_world.tolist(),
        "max_world": converter.max_world.tolist(),
    }
    metadata["sun_model"] = {
        "coef_": sun_A.tolist(),
        "intercept_": sun_b.tolist(),
        "sun_dir_ecef": sun_dir_ecef.tolist(),
        "camera_to_sun": myM.tolist(),
    }

    return metadata


def pipeline_pan(metadata, converter, model_W2V_coef_, model_W2V_intercept_):
    # check for the pan image
    # now we do the MSI part :
    img_name = metadata["img"]
    img_path = os.path.join(converter.pan_dir, img_name)
    try:
        pan_image = rasterio.open(img_path).read().transpose(1, 2, 0)
    except:
        pan_image = iio.read(os.path.join(converter.pan_dir, img_name))
    scale_factor_h = metadata["height"] // pan_image.shape[0]
    scale_factor_w = metadata["width"] // pan_image.shape[1]
    check_scale_factor(
        scale_factor_h=scale_factor_h,
        scale_factor_w=scale_factor_w,
        metadata=metadata,
        img=pan_image,
        img_name=img_name,
    )
    if converter.change_resolution:
        if (
            metadata["height"] != pan_image.shape[0]
            or metadata["width"] != pan_image.shape[1]
        ):
            print(
                f"Changing resolution of the pan  metadata from {metadata['height']}x{metadata['width']} to {pan_image.shape[0]}x{pan_image.shape[1]}"
            )
        metadata["height"] = pan_image.shape[0]
        metadata["width"] = pan_image.shape[1]
    #########################################################
    ### Computing the affine approximation of the sun
    #########################################################
    # First we compute the change of basis M from lonlat to world coordinates
    # and the center of the scene in world coords
    # M, centerofscene_ECEF = converter.computeJacobian()

    M = np.eye(3)
    centerofscene_UTM = np.zeros(3)

    # Then we compute the sun direction in ECEF
    local_sun_direction = -get_dir_vec_from_el_az(
        elevation_deg=90 - float(metadata["sun_elevation"]),
        azimuth_deg=float(metadata["sun_azimuth"]),
    )
    sun_dir_ecef = M.T @ local_sun_direction
    sun_dir_ecef = sun_dir_ecef / (model_W2V_coef_ @ sun_dir_ecef)[2]

    # Then we compute the affine model for the sun
    Asun_dir_ecef = model_W2V_coef_ @ sun_dir_ecef
    myM = np.array([[1, 0, -Asun_dir_ecef[0]], [0, 1, -Asun_dir_ecef[1]], [0, 0, 1]])
    sun_A = myM @ model_W2V_coef_
    sun_b = (
        -sun_A @ centerofscene_UTM
        + model_W2V_coef_ @ centerofscene_UTM
        + model_W2V_intercept_
    )

    # Enrich the metadata with the new models
    metadata["centerofscene_UTM"] = centerofscene_UTM.tolist()
    metadata["virtual_camera"] = False
    metadata["model"] = {
        "coef_": model_W2V_coef_.tolist(),
        "intercept_": model_W2V_intercept_.tolist(),
        "scale": converter.scale,
        "n": converter.n,
        "l": converter.l,
        # "rotation": converter.R.tolist(),
        "center": converter.shift.tolist(),
        "min_world": converter.min_world.tolist(),
        "max_world": converter.max_world.tolist(),
    }
    metadata["sun_model"] = {
        "coef_": sun_A.tolist(),
        "intercept_": sun_b.tolist(),
        "sun_dir_ecef": sun_dir_ecef.tolist(),
        "camera_to_sun": myM.tolist(),
    }
    return metadata


def pipeline(metadata, converter):
    rpc = rpcm.RPCModel(d=metadata["rpc"], dict_format="rpcm")
    #  store the metadata for pan and msi metadatas

    #########################################################
    ### Compute the affine approximation of the camera model
    #########################################################
    model_W2V = approximate_W2V_affine(
        rpc=rpc,
        width=metadata["width"],
        height=metadata["height"],
        min_altitude=metadata["min_alt"],
        max_altitude=metadata["max_alt"],
        converter=converter,
    )

    A = np.array(model_W2V.coef_)
    b = np.array(model_W2V.intercept_)

    # beware the deepcopy
    # run pipeline on the msi images :

    msi_metadata = pipeline_msi(
        metadata=deepcopy(metadata),
        converter=converter,
        model_W2V_coef_=A,
        model_W2V_intercept_=b,
    )

    # run pipeline on the pan images
    pan_metadata = pipeline_pan(
        metadata=deepcopy(metadata),
        converter=converter,
        model_W2V_coef_=A,
        model_W2V_intercept_=b,
    )

    return msi_metadata, pan_metadata


def create_nadir_cam(output_metadatas):
    # Generate a metadata for a perfectly nadir camera
    for key in output_metadatas.keys():
        metadata = deepcopy(output_metadatas[key][0])
        metadata["img"] = "Nadir"
        metadata["model"]["coef_"] = [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, metadata["model"]["scale"]],
        ]
        metadata["model"]["intercept_"] = [0.0, 0.0, 0.0]
        metadata["virtual_camera"] = True
        output_metadatas[key].append(metadata)

    return output_metadatas


def main(
    root_dir: str = "~/data/satnerf/IARPA/root_dir/rpcs_ba",
    scene_name: str = "IARPA_001",
    dataset_destination_path: str = "~/data/satnerf/affine_models",
    save_data: bool = True,
    prefix: str = "normalized_v6",
    change_resolution: bool = True,
    msi_dir="/workspaces/external_datasets/satnerf/DFC2019/track3-PMSI-crops/msi-crops",
    pan_dir="/workspaces/external_datasets/satnerf/DFC2019/track3-PMSI-crops/pan-crops",
    cfg: "omegaconf.DictConfig" = None,
    output_msi: bool = True,
    output_pan: bool = True,
    path_ply: str = None,
    traintest_dir="/workspaces/external_datasets/satnerf/root_dir/traintest_dir",
):
    assert scene_name in [
        "JAX_004",
        "JAX_068",
        "JAX_214",
        "JAX_260",
        "IARPA_001",
        "IARPA_002",
        "IARPA_003",
    ], f"scene_name not in the list of available scenes, got {scene_name}"
    SCENE_METADATA = Path(os.path.expanduser(os.path.join(root_dir, scene_name)))
    DATASET_DESTINATION = Path(
        os.path.expanduser(
            os.path.join(dataset_destination_path, f"{scene_name}_{prefix}")
        )
    )
    traintest_dir = os.path.expanduser(os.path.join(traintest_dir, scene_name))
    MSI_DIR = os.path.expanduser(os.path.join(msi_dir, scene_name))
    PAN_DIR = os.path.expanduser(os.path.join(pan_dir, scene_name))
    print("loading the MSI images from ", MSI_DIR, " and the PAN images from ", PAN_DIR)
    print("SCENE METADATA is ", SCENE_METADATA)
    # Read the scene and for each image:
    # 1. Open the corresponding .json metadata file
    metadatas = sorted(glob(f"{SCENE_METADATA}/*.json"))
    metadatas = map(open_json_file, metadatas)
    metadatas = list(metadatas)
    assert (
        metadatas != []
    ), f"No metadata files found in the scene directory, we looked at folder:{SCENE_METADATA}"

    T = PanConverter(
        scene_metadatas=metadatas,
        MSI_DIR=MSI_DIR,
        PAN_DIR=PAN_DIR,
        change_resolution=change_resolution,
    )

    # output arguments
    output_metadatas = {"pan": [], "msi": []}
    # 2. Run the conversion pipeline
    for metadata in metadatas:
        msi_metadata, pan_metadata = pipeline(metadata=metadata, converter=T)
        # add the metadata to the output
        if output_msi:
            output_metadatas["msi"].append(msi_metadata)
        if output_pan:
            output_metadatas["pan"].append(pan_metadata)
    # metadatas = map(lambda m: pipeline(m, T), metadatas)
    # metadatas = list(metadatas)
    # if path_ply is provided, recenter the model around the ply
    if path_ply is not None:
        print("you provided a ply file, recentring the model around the ply")
        xyz, rgb = T.transform_pcd(path_ply)
    else:
        xyz = None
        rgb = None
    # Generate a metadata for a perfectly nadir camera
    output_metadatas = create_nadir_cam(output_metadatas)
    # Now that the new metadata has been computed, we run a few tests
    test(output_metadatas["msi"])
    if save_data:
        # Finally, we save the new metadata
        os.makedirs(DATASET_DESTINATION, exist_ok=True)
        # save the json
        with open(f"{DATASET_DESTINATION}/affine_models.json", "w") as f:
            json.dump(output_metadatas, f, indent=4)
        # Copy also the test.txt and train.txt files
        os.system(f"cp {traintest_dir}/test.txt {DATASET_DESTINATION}")
        os.system(f"cp {traintest_dir}/train.txt {DATASET_DESTINATION}")
        print("we saved in ", DATASET_DESTINATION)
        if not isinstance(cfg, type(None)):
            with open(os.path.join(DATASET_DESTINATION, "cfg.yaml"), "w") as f:
                OmegaConf.save(cfg, f)

        if not isinstance(xyz, type(None)):
            storePly(
                xyz=xyz,
                path=os.path.join(DATASET_DESTINATION, "input_pcd.ply"),
                rgb=rgb,
            )
    else:
        print("Not saving the data, just testing the pipeline")
    return output_metadatas


@hydra.main(
    version_base="1.2",
    config_path="config",
    config_name="main_affine.yaml",
)
def hydra_main(cfg):
    # This function is used to run the main function with hydra
    # It allows to use the command line interface to run the script
    # and to pass the configuration file
    output_metadata = main(
        root_dir=cfg.root_dir,
        scene_name=cfg.scene_name,
        dataset_destination_path=cfg.dataset_destination_path,
        save_data=cfg.save_data,
        prefix=cfg.prefix,
        change_resolution=cfg.change_resolution,
        msi_dir=cfg.msi_dir,
        pan_dir=cfg.pan_dir,
        traintest_dir=cfg.traintest_dir,
        cfg=cfg,
        path_ply=cfg.get("path_ply", None),
    )
    return output_metadata


if __name__ == "__main__":
    # tyro is only used for CLI
    hydra_main()
