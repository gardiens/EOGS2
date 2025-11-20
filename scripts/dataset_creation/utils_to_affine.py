import json
import numpy as np
import rpcm
import sklearn.linear_model as lm
from .converter import MyConverter


def check_scale_factor(scale_factor_h, scale_factor_w, metadata, img, img_name):
    assert (
        round(scale_factor_h) == round(scale_factor_w)
    ), "Scale factors must be equal for MSI image, got {} and {} for given rpc width {} and height {}, image width {} height {} and image name {}".format(
        scale_factor_h,
        scale_factor_w,
        metadata["width"],
        metadata["height"],
        img.shape[1],
        img.shape[0],
        img_name,
    )
    return


def open_json_file(file_path):
    with open(file_path, "r") as f:
        file = json.load(f)
    return file


def test(metadatas):
    for i in range(len(metadatas)):
        for j in range(len(metadatas)):
            A = metadatas[i]["model"]["coef_"]
            sundir = metadatas[j]["sun_model"]["sun_dir_ecef"]
            A = np.array(A)
            sundir = np.array(sundir)
            lolz = (A @ sundir)[2]
            assert abs(lolz - 1) < 1e-4, lolz
    # assert the nadir camera is in the metadats
    assert "Nadir" in [
        metadatas[i]["img"] for i in range(len(metadatas))
    ], "Nadir camera not found in metadatas, got {}".format(
        [metadatas[i]["img"] for i in range(len(metadatas))]
    )


#### Compute sun direction
def get_dir_vec_from_el_az(elevation_deg, azimuth_deg):
    # convention: elevation is 0 degrees at nadir, 90 at frontal view
    el = np.radians(90 - elevation_deg)
    az = np.radians(azimuth_deg)
    dir_vec = -1.0 * np.array(
        [np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)]
    )
    # dir_vec = dir_vec / np.linalg.norm(dir_vec)
    return dir_vec


def approximate_W2V_affine(
    rpc: rpcm.RPCModel,
    width: int,
    height: int,
    min_altitude: float,
    max_altitude: float,
    converter: MyConverter,
):
    # crate a meshgrid of the image
    # This defines the view/local/intrinsic coordinate system of the current image
    Nu = 31
    Nv = 37
    Na = 29
    u = np.linspace(0, width - 1, Nu)
    v = np.linspace(0, height - 1, Nv)
    a = np.linspace(min_altitude, max_altitude, Na)
    U, V, A = np.meshgrid(u, v, a, indexing="ij")
    UVA = np.stack([U, V, A], axis=-1)

    # This are the view/local/intrinsic coordinates of the image with U and V in [-1,1] and A in [min_altitude, max_altitude]
    view = (UVA + np.array([0.5, 0.5, 0])) * np.array([1 / width, 1 / height, 1])
    view[..., :2] = view[..., :2] * 2 - 1

    # Now we want to compute a world/global/extrinsic coordinate system for all the images in the scene
    # We do this by computing the lonlat coordinates (this is already a global coordinate system)
    # Then we use a custom converter that should be global, in the sense that it should be the same for all images in the scene.
    LON, LAT = rpc.localization(U.flatten(), V.flatten(), A.flatten())
    LON = LON.reshape(Nu, Nv, Na)
    LAT = LAT.reshape(Nu, Nv, Na)
    world_coords = converter.LONLAT2world(np.stack([LON, LAT, A], axis=-1))

    # Now we learn a linear mapping where:
    # - inputs are the world/global coordinates
    # - outputs are view/local UVA coordinates
    model_W2V = lm.LinearRegression(fit_intercept=True)
    model_W2V.fit(X=world_coords.reshape(-1, 3), y=view.reshape(-1, 3))

    return model_W2V
