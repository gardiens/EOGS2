#  Really ugly code where  I put all the typing stuff in one file, do not write actual code here pls!
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace
    from scene.cameras import AffineCamera
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers.dataset_affine import AffineCameraInfo, AffineCamera
    from arguments import ModelParams, CameraParams, ClearmlParams,PipelineParams
    from typing import List, Dict, Any, Union
    from scene.MS_scene import  MSScene

    from scene.dataset_readers import sceneLoadTypeCallbacks
