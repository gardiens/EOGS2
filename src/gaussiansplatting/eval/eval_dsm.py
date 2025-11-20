import sys
import os

# add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dsmr import compute_shift, apply_shift
import hydra

import iio
import rasterio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rasterio.io import MemoryFile
    from omegaconf import DictConfig
import numpy as np
import os

from typing import Optional
# ruff: noqa: F722
try:
    import clearml
    try:
        from clearml_utils import safe_resume_clearml, safe_init_clearml
    except:
        from utils.clearml_utils import safe_resume_clearml, safe_init_clearml
    CLEARML_FOUND = True 
except:
    CLEARML_FOUND = False


from tyro.conf import FlagConversionOff


def mask_dsm(dsm, water_mask, vis_mask, tree_mask):
    # all the gt_dsm was before pred_dsm but I didn't check if it works.
    if water_mask is not None:
        water_mask = water_mask[: dsm.shape[0], : dsm.shape[1]]
        dsm[water_mask] = np.nan

    # Use visibility mask if available
    if vis_mask is not None:
        dsm[vis_mask] = np.nan

    # use tree mask if available
    if tree_mask is not None:
        if dsm.shape != tree_mask.shape:
            print(
                "warning, dsm and tree_mask have different shapes, Cropping the dsm, it should only happen for IARPA 002  "
            )
            dsm = dsm[: tree_mask.shape[0], : tree_mask.shape[1]]
        dsm[np.logical_not(tree_mask)] = np.nan
    return dsm


def dsm_pointwise_diff(
    pred_dsm: str,
    gt_dsm: "np.array",
):
    # register and compute mae
    transform = compute_shift(gt_dsm, pred_dsm, scaling=False)
    pred_rdsm = apply_shift(pred_dsm, *transform)
    h = min(pred_rdsm.shape[0], gt_dsm.shape[0])
    w = min(pred_rdsm.shape[1], gt_dsm.shape[1])
    max_gt_alt = gt_dsm.max()
    min_gt_alt = gt_dsm.min()
    pred_rdsm = np.clip(pred_rdsm, min_gt_alt - 10, max_gt_alt + 10)
    diff = pred_rdsm[:h, :w] - gt_dsm[:h, :w]
    return diff, pred_rdsm


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


def load_mae_things(
    gt_dir: str, aoi_id: str, enable_vis_mask: bool = True, filter_tree: bool = True
):
    """in_dsm_path is a string with the path to the NeRF generated dsm gt_dsm_path is a string with
    the path to the reference lidar dsm bbx_metadata is a 4-valued array with format (x, y, s, r)
    where [x, y] = offset of the dsm bbx, s = width = height, r = resolution (m per pixel)"""

    gt_dsm_path = os.path.join(gt_dir, "{}_DSM.tif".format(aoi_id))
    print("we are loading the gt_dsm_path", gt_dsm_path)
    # if a v2 exists, use it
    if os.path.exists(os.path.join(gt_dir, "{}_CLS_v2.tif".format(aoi_id))):
        gt_seg_path = os.path.join(gt_dir, "{}_CLS_v2.tif".format(aoi_id))
    else:
        gt_seg_path = os.path.join(gt_dir, "{}_CLS.tif".format(aoi_id))

    assert os.path.exists(gt_dsm_path), f"{gt_dsm_path} not found"
    assert os.path.exists(gt_seg_path), f"{gt_seg_path} not found"

    # Check whether a txt file exists. If so, use it
    # Otherwise assume that the DSM is geolocalized
    if os.path.exists(os.path.join(gt_dir, "{}_DSM.txt".format(aoi_id))):
        # Mostly DFC2019 scenes
        gt_roi_path = os.path.join(gt_dir, "{}_DSM.txt".format(aoi_id))
        print("Using gt_roi_path", gt_roi_path)
        gt_roi_metadata = np.loadtxt(gt_roi_path)
    else:
        # mostly IARPA scenes
        src = rasterio.open(gt_dsm_path)
        gt_roi_metadata = np.array(
            [src.bounds.left, src.bounds.bottom, min(src.height, src.width), src.res[0]]
        )
        del src
    # read gt dsm
    with rasterio.open(gt_dsm_path, "r") as f:
        gt_dsm = f.read()[0, :, :]
    if enable_vis_mask:
        vis_mask_path = os.path.join(
            os.path.dirname(__file__), f"vis_masks/{aoi_id}.tif".format(aoi_id)
        )
        if not os.path.exists(vis_mask_path):
            vis_mask = None
        else:
            vis_mask = rasterio.open(vis_mask_path).read()[0, ...] > 0.5
    else:
        vis_mask = None

    if filter_tree:
        tree_mask_path = os.path.join(
            os.path.dirname(__file__), f"tree_masks/{aoi_id}.png".format(aoi_id)
        )
        if not os.path.exists(tree_mask_path):
            tree_mask = None
        else:
            tree_mask = rasterio.open(tree_mask_path).read()[0, ...] > 0.5
    else:
        tree_mask = None

    if gt_seg_path is not None:
        with rasterio.open(gt_seg_path, "r") as f:
            mask = f.read()[0, :, :]
            water_mask = mask.copy()
            water_mask[mask != 9] = 0
            water_mask[mask == 9] = 1
            water_mask = water_mask.astype(bool)
        if ("CLS.tif" in gt_seg_path) and (
            os.path.exists(gt_seg_path.replace("CLS.tif", "WATER.png"))
        ):
            # print("found alternative water mask!")
            mask = iio.read(gt_seg_path.replace("CLS.tif", "WATER.png"))[..., 0]
            water_mask = mask == 0
    else:
        water_mask = None
    # load the corresponding in_dsm crop

    return gt_roi_metadata, gt_dsm, water_mask, vis_mask, tree_mask


def compute_mae(
    pred_dsm_path: str,
    gt_dir: str,
    aoi_id: str,
    enable_vis_mask: bool = True,
    filter_tree: bool = True,
):
    mae_compute = Mae_Computer(
        gt_dir=gt_dir,
        aoi_id=aoi_id,
        enable_vis_mask=enable_vis_mask,
        filter_tree=filter_tree,
    )
    mae, diff, rdsm, profile, _ = mae_compute.compute_mae_from_path(
        pred_dsm_path=pred_dsm_path
    )

    return mae, diff, rdsm, profile


import matplotlib.pyplot as plt


def save_dsm_diff(out_dir, aoi_id, mae, profile, rdsm, diff, prefix: str = ""):
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        rdsm_diff_path = os.path.join(out_dir, "{}_rdsm_diff.tif".format(aoi_id))
        with rasterio.open(rdsm_diff_path, "w", **profile) as dst:
            dst.write(diff, 1)
        abs_diff = np.abs(diff)
        abs_rdsm_diff_path = os.path.join(
            out_dir, "{}_rdsm_abs_diff.tif".format(aoi_id)
        )
        with rasterio.open(abs_rdsm_diff_path, "w", **profile) as dst:
            dst.write(abs_diff, 1)
        png_out_dir = os.path.join(os.path.dirname(out_dir), "png")

        # show the abs_diff as a png
        abs_diff_png_path = os.path.join(
            png_out_dir, "{}_rdsm_abs_diff.png".format(aoi_id)
        )
        os.makedirs(os.path.dirname(abs_diff_png_path), exist_ok=True)
        print(" on passe par la?")
        fig, ax = plt.subplots(figsize=(5, 5))

        im = ax.imshow(abs_diff, cmap="viridis")
        fig.colorbar(im, ax=ax)
        ax.set_title(prefix + "absolute rdsm difference")

        plt.savefig(abs_diff_png_path)

        plt.close()
        # show the rdsm as a png
        rdsm_png_path = os.path.join(png_out_dir, "{}_rdsm.png".format(aoi_id))
        os.makedirs(os.path.dirname(rdsm_png_path), exist_ok=True)
        plt.imshow(rdsm, cmap="viridis")
        plt.colorbar()
        plt.title(prefix + f" rdsm for mae {mae:.2f}")
        plt.savefig(rdsm_png_path)
        plt.show()
        # report this with clearml as a plot
        plt.close()

        rdsm_path = os.path.join(out_dir, "{}_rdsm.tif".format(aoi_id))
        with rasterio.open(rdsm_path, "w", **profile) as dst:
            dst.write(rdsm, 1)


def compute_mae_and_save_dsm_diff(
    pred_dsm_path: str,
    gt_dir: str,
    aoi_id: str,
    out_dir: Optional[str] = None,
    enable_vis_mask: bool = True,
    filter_tree: bool = True,
    prefix="",
) -> float:
    mae, diff, rdsm, profile = compute_mae(
        pred_dsm_path,
        gt_dir,
        aoi_id,
        enable_vis_mask=enable_vis_mask,
        filter_tree=filter_tree,
    )
    # save the dsm
    save_dsm_diff(
        out_dir=out_dir,
        aoi_id=aoi_id,
        mae=mae,
        profile=profile,
        rdsm=rdsm,
        diff=diff,
        prefix=prefix,
    )

    return mae


class Mae_Computer:
    def __init__(
        self,
        gt_dir: str,
        aoi_id: str,
        enable_vis_mask: bool = True,
        filter_tree: bool = True,
    ):
        gt_dsm, water_mask, vis_mask, tree_mask, ulx, uly, lrx, lry = self.preprocess(
            gt_dir, aoi_id, enable_vis_mask, True
        )
        self.tree_mask = tree_mask
        self.gt_dsm = gt_dsm
        if not filter_tree:
            tree_mask = None
        self.gt_dsm = mask_dsm(
            dsm=gt_dsm, water_mask=water_mask, vis_mask=vis_mask, tree_mask=tree_mask
        )

        self.ulx = ulx
        self.uly = uly
        self.lrx = lrx
        self.lry = lry

        # hacky things in case you wnat to mask at some point with trees
        self._gt_dsm_masked = None
        return

    def preprocess(self, gt_dir, aoi_id, enable_vis_mask, filter_tree):
        gt_roi_metadata, gt_dsm, water_mask, vis_mask, tree_mask = load_mae_things(
            gt_dir=gt_dir,
            aoi_id=aoi_id,
            enable_vis_mask=enable_vis_mask,
            filter_tree=filter_tree,
        )

        xoff, yoff = gt_roi_metadata[0], gt_roi_metadata[1]
        xsize, ysize = int(gt_roi_metadata[2]), int(gt_roi_metadata[2])
        resolution = gt_roi_metadata[3]

        # define projwin for gdal translate
        ulx, uly, lrx, lry = (
            xoff,
            yoff + ysize * resolution,
            xoff + xsize * resolution,
            yoff,
        )
        return gt_dsm, water_mask, vis_mask, tree_mask, ulx, uly, lrx, lry

    def pred_dsm_rasterio(self, f):
        profile = f.profile
        transform = f.transform

        # Compute the window corresponding to the window
        ulx, uly = ~transform * (self.ulx, self.uly)
        lrx, lry = ~transform * (self.lrx, self.lry)
        window = rasterio.windows.Window(ulx, uly, lrx - ulx, lry - uly)

        pred_dsm = f.read(1, window=window)

        profile.update(
            height=lry - uly, width=lrx - ulx, transform=f.window_transform(window)
        )
        return pred_dsm, profile

    def load_pred_dsm_from_path(self, pred_dsm_path: str):
        """Load the dsm to the disk and return the pred_dsm and profile."""
        with rasterio.open(pred_dsm_path, "r") as f:
            pred_dsm, profile = self.pred_dsm_rasterio(f=f)

        return pred_dsm, profile

    def compute_mae_from_memfile(self, memfile: "MemoryFile"):
        with memfile.open() as f:
            pred_dsm, profile = self.pred_dsm_rasterio(f=f)
        diff, rdsm = dsm_pointwise_diff(
            pred_dsm=pred_dsm,
            gt_dsm=self.gt_dsm,
        )
        mae = self._compute_mae(diff)
        return mae, diff, rdsm, profile

    def _compute_mae(self, diff):
        # check if all the value of diff are nan
        mae = np.nanmean(abs(diff.ravel()))
        if np.isnan(mae):
            raise ValueError(
                "The computed MAE is NaN, this means that the diff array contains only NaN values. Probably either a jit fastmath issue or a problem with the rpc cameras"
            )
        return mae

    def remove_dsm(self, pred_dsm_path: str):
        """Remove the dsm file from the disk."""
        if os.path.exists(pred_dsm_path):
            os.remove(pred_dsm_path)
        else:
            print(f"{pred_dsm_path} does not exist, cannot remove it.")
        return

    def get_gt_dsm(self, force_use_tree_mask: bool = False):
        if force_use_tree_mask:
            tree_mask = self.tree_mask
            if self._gt_dsm_masked is None:
                self._gt_dsm_masked = mask_dsm(
                    dsm=self.gt_dsm.copy(),
                    water_mask=None,
                    vis_mask=None,
                    tree_mask=tree_mask,
                )
            return self._gt_dsm_masked
        else:
            return self.gt_dsm

    def compute_mae_from_path(
        self, pred_dsm_path: str, force_use_tree_mask: bool = False
    ):
        pred_dsm, profile = self.load_pred_dsm_from_path(pred_dsm_path)
        gt_dsm = self.get_gt_dsm(force_use_tree_mask=force_use_tree_mask)

        diff, rdsm = dsm_pointwise_diff(
            pred_dsm=pred_dsm,
            gt_dsm=gt_dsm,
        )
        mae = self._compute_mae(diff)
        return mae, diff, rdsm, profile, pred_dsm

    def compute_mae_from_pred_dsm(
        self, pred_dsm, profile, force_use_tree_mask: bool = False
    ):
        gt_dsm = self.get_gt_dsm(force_use_tree_mask=force_use_tree_mask)
        diff, rdsm = dsm_pointwise_diff(
            pred_dsm=pred_dsm,
            gt_dsm=gt_dsm,
        )
        mae = np.nanmean(abs(diff.ravel()))
        return mae, diff, rdsm, profile


def main(
    pred_dsm_path: str,
    gt_dir: str,
    aoi_id: str,
    out_dir: Optional[str] = None,
    enable_vis_mask: bool = True,
    filter_tree: bool = True,
    resume_clearml: "FlagConversionOff[bool]" = True,
    task_name: str = "JAX_068_normalized_v6",
    debug: bool = False,
    prefix: str = "",
):

    if not debug and CLEARML_FOUND:
        if resume_clearml:
            task = safe_resume_clearml(project_name="EOGS", task_name=task_name)
        else:
            task = safe_init_clearml(project_name="EOGS", task_name=task_name)
        print("who is task?", task)

    if not pred_dsm_path.endswith(".iio"):
        print("WARNING: pred_dsm_path should be a .iio file,got ", pred_dsm_path)
        # in this case replace by .iio
        pred_dsm_path = pred_dsm_path.replace(pred_dsm_path.split(".")[-1], "iio")
    mae = compute_mae_and_save_dsm_diff(
        pred_dsm_path=pred_dsm_path,
        gt_dir=gt_dir,
        aoi_id=aoi_id,
        out_dir=out_dir,
        enable_vis_mask=enable_vis_mask,
        filter_tree=filter_tree,
        prefix=prefix,
    )
    print("MAE:", mae)
    if not debug and CLEARML_FOUND:
        task.get_logger().report_scalar("MAE", "final_MAE", value=mae, iteration=0)
        task.close()
    return mae




def get_half_dsm_out_dir(cfg_dsm, cfg):
    """Get the half_dsm and out dir from the cfg_dsm and cfg.

    The subtelty is if the iteration in the rendering config is -1
    """
    num_iterations = cfg.get("numiterations", None)
    model_path = cfg_dsm.get("model_path", None)
    iteration = cfg.get("iteration", None)
    if iteration != -1:
        return cfg_dsm.out_dir, cfg_dsm.half_dsm_path
    else:
        max_itr = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
        half_dsm_path = os.path.join(
            model_path, "test_opNone", f"ours_{max_itr}", "dsm"
        )
        out_dir = os.path.join(model_path, "test_opNone", f"ours_{max_itr}", "rdsm")
        return out_dir, half_dsm_path


# def _main_hydra_dsm(cfg:"DictConfig"):


@hydra.main(
    version_base="1.2", config_path="../gs_config", config_name="rendering.yaml"
)
def main_hydra_dsm(cfg: "DictConfig"):
    cfg_dsm = cfg.get("eval", None)
    half_dsm_path = cfg_dsm.get("half_dsm_path", None)
    out_dir = cfg_dsm.get("out_dir", None)
    out_dir, half_dsm_path = get_half_dsm_out_dir(cfg_dsm, cfg)
    print("the out dir is ", out_dir)
    if half_dsm_path is None:
        raise ValueError(
            "half_dsm_path must be provided in the config,got None for cfg {}".format(
                cfg_dsm
            )
        )
    # get the pred dsm path
    name_nadir = cfg_dsm.nadir_name
    if cfg_dsm.get("pred_dsm_path", None):
        print("using pred_dsm_path from config")
        pred_dsm_path = cfg_dsm.pred_dsm_path
    else:
        pred_dsm_path = os.path.join(half_dsm_path, name_nadir)
    scene_name = cfg_dsm.scene_name
    gt_dir = cfg_dsm.gt_dir
    aoi_id = cfg_dsm.aoi_id
    print("we will use the pred_dsm_path", pred_dsm_path)
    mae = main(
        pred_dsm_path=pred_dsm_path,
        gt_dir=gt_dir,
        aoi_id=aoi_id,
        out_dir=out_dir,
        enable_vis_mask=cfg_dsm.get("enable_vis_mask", True),
        filter_tree=cfg_dsm.get("filter_tree", True),
        resume_clearml=cfg_dsm.get("resume_clearml"),
        task_name=cfg_dsm.get("task_name"),
        debug=cfg_dsm.get("debug", False),
        prefix=cfg_dsm.get("prefix", ""),
    )
    return mae


if __name__ == "__main__":
    import tyro

    tyro.cli(
        main,
    )
