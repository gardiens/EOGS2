from plyflatten import plyflatten
from plyflatten.utils import rasterio_crs, crs_proj
import affine
import numpy as np


def compute_dsm_from_view(view, rendered_uva, scene_params, scene_name):
    cloud = view.UVA_to_ECEF(rendered_uva.detach().reshape((-1, 3))).cpu().numpy()

    # Unnormalized the point cloud so we're in normal utm again
    cloud = cloud * scene_params[1] + scene_params[0]

    # TODO: Each sceneparams has its own meter/pixel resolution
    if "IARPA" in scene_name:
        resolution = 0.3
    elif "JAX" in scene_name:
        resolution = 0.5
    else:
        raise ValueError("Unknown sceneparams,got {}.".format(scene_name))
    xmin, xmax = cloud[:, 0].min(), cloud[:, 0].max()
    ymin, ymax = cloud[:, 1].min(), cloud[:, 1].max()
    xoff = np.floor(xmin / resolution) * resolution
    xsize = int(1 + np.floor((xmax - xoff) / resolution))
    yoff = np.ceil(ymax / resolution) * resolution
    ysize = int(1 - np.floor((ymin - yoff) / resolution))

    # run plyflatten
    dsm = plyflatten(
        cloud,
        xoff,
        yoff,
        resolution,
        xsize,
        ysize,
        radius=1,
        sigma=float("inf"),
    )
    crs = rasterio_crs(
        crs_proj("{}{}".format(scene_params[2], scene_params[3]), crs_type="UTM")
    )

    profile = {}
    profile["dtype"] = dsm.dtype
    profile["height"] = dsm.shape[0]
    profile["width"] = dsm.shape[1]
    profile["count"] = 1
    profile["driver"] = "GTiff"
    profile["nodata"] = float("nan")
    profile["crs"] = crs
    profile["transform"] = affine.Affine(resolution, 0.0, xoff, 0.0, -resolution, yoff)
    return profile, dsm
