from omegaconf import DictConfig
import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)  #!
from train_pan import *


@hydra.main(version_base="1.2", config_path="gs_config", config_name="rendering.yaml")
def main(cfg: DictConfig) -> None:
    print("for debug", cfg.model)
    mp = ModelParams(cfg.get("model", None))
    op = OptimizationParams(cfg.get("optimization", None))
    cmlparams = ClearmlParams(cfg.get("clearml", None))
    pp: "PipelineParams" = PipelineParams(cfg=cfg.get("pipeline", None))
    print("for debug", mp.model_path)


if __name__ == "__main__":
    main()
    main()
    # To run this script, use the command:
    # python test_hydra.py
    # Ensure that the config directory and main_affine.yaml file exist in the specified path.
