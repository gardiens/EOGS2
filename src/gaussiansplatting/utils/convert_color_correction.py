from utils.camera_utils import get_list_cam
import torch
from typing import TYPE_CHECKING
from torch import nn

if TYPE_CHECKING:
    from utils.typing_utils import *
from torch import nn


class base_cc_train_to_test:
    def __init__(self):
        pass

    def get_train_test_camlist(self, train_viewpoints, test_cameras, opt):
        list_train_cameras = []
        list_test_cameras = []
        list_cam_int = get_list_cam(train_viewpoints[0], opt)
        list_train_cameras = [[] for _ in range(len(list_cam_int))]
        list_test_cameras = [[] for _ in range(len(list_cam_int))]

        for train_viewpoint in train_viewpoints:
            list_cam: "List[AffineCamera]" = get_list_cam(train_viewpoint, opt)
            for k in range(len(list_cam)):
                list_train_cameras[k].append(list_cam[k])
        for test_viewpoint in test_cameras:
            list_cam: "List[AffineCamera]" = get_list_cam(test_viewpoint, opt)
            for k in range(len(list_cam)):
                list_test_cameras[k].append(list_cam[k])
        return list_train_cameras, list_test_cameras

    def perform_cc_to_test(self, train_viewpoints, test_cameras, opt):
        list_train_cameras, list_test_cameras = self.get_train_test_camlist(
            train_viewpoints, test_cameras, opt
        )

        for k in range(len(list_train_cameras)):
            self.correct_test_cameras(list_train_cameras[k], list_test_cameras[k])
        return test_cameras

    def correct_test_cameras(
        self, train_cameras: "List[AffineCamera]", test_cameras: "List[AffineCamera]"
    ):
        raise NotImplementedError


class perform_cc_by_ref_test(base_cc_train_to_test):
    def __init__(self):
        pass

    def correct_test_cameras(
        self, train_cameras: "List[AffineCamera]", test_cameras: "List[AffineCamera]"
    ):
        view = None
        for train_cam in train_cameras:
            if train_cam.is_reference_camera:
                view = train_cam
        if view is None:
            raise ValueError("No reference camera found in train cameras")

        for cam in test_cameras:
            cam.color_correction.weight = nn.Parameter(
                view.color_correction.weight.clone()
            )
            cam.color_correction.bias = nn.Parameter(view.color_correction.bias.clone())


from torch import nn


class perform_average_cc_test(base_cc_train_to_test):
    def __init__(self):
        pass

    def correct_test_cameras(self, train_cameras, test_cameras):
        # Compute average color correction from train cameras
        avg_weight = torch.zeros_like(train_cameras[0].color_correction.weight)
        avg_bias = torch.zeros_like(train_cameras[0].color_correction.bias)
        for cam in train_cameras:
            avg_weight += cam.color_correction.weight
            avg_bias += cam.color_correction.bias
        avg_weight /= len(train_cameras)
        avg_bias /= len(train_cameras)

        # Apply average color correction to test cameras
        for cam in test_cameras:
            cam.color_correction.weight = nn.Parameter(avg_weight.clone())
            cam.color_correction.bias = nn.Parameter(avg_bias.clone())
        return test_cameras


class perform_closesttime_cc_test(base_cc_train_to_test):
    def __init__(self):
        raise NotImplementedError("Because JAX doesn't have timestamp yet ")


def load_convert_color_correction_train_to_test(name: str = "None"):
    if name == "ref":
        return perform_cc_by_ref_test()
    elif name == "average":
        return perform_average_cc_test()
    elif name == "closesttime":
        return perform_closesttime_cc_test()
    else:
        raise NotImplementedError(f"color correction {name} not implemented")
