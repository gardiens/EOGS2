from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra


def init_config(config_name="train.yaml", overrides=[], config_path="gs_config"):
    # Registering the "eval" resolver allows for advanced config
    # interpolation with arithmetic operations:
    # https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    GlobalHydra.instance().clear()
    with initialize(version_base="1.2", config_path=config_path):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg
