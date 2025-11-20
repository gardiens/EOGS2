from utils.hydra_utils import init_config
from train_pan import *


def test_init():
    cfg = init_config(
        config_name="train.yaml",
        config_path="../gs_config",
    )
    cfg
    opt = OptimizationParams(cfg.get("optimization", None))
    cmlparams = ClearmlParams(cfg.get("clearml", None))
    pipe: "PipelineParams" = PipelineParams(cfg=cfg.get("pipeline", None))
    test_iterations_default = (
        list(range(0, 100)) + list(range(100, 1000, 10)) + list(range(1000, 10000, 50))
    )
    test_iterations_default = sorted(list(set(test_iterations_default)))
    test_iterations_default = []
    sceneparams = ModelParams(cfg=cfg.model)
    sceneparams.load_pan
    gaussians = GaussianModel(sceneparams.sh_degree)

    from scene.MS_scene import MSScene

    scene = MSScene(sceneparams, gaussians)

    viewpoint_stack = None
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
    scene.getTrainCameras()[0]
