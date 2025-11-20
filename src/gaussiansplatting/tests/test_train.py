from train_pan import *
from utils.hydra_utils import init_config
import pytest

from train_pan import main as main_train
from render_pan import main as main_render
from eval.eval_dsm import main_hydra_dsm as main_eval_dsm
from tsdf import main_hydra_tsdf as main_tsdf


@pytest.mark.order(0)
def test_affine():
    cfg = init_config(
        config_name="main_affine.yaml",
        overrides=["prefix=pytest"],
        config_path="../../../scripts/dataset_creation/config/",
    )
    # should be thuis path /workspaces/EOGS/scripts/dataset_creation/config
    from scripts.dataset_creation.to_affine import hydra_main as hydra_main

    hydra_main(cfg)


@pytest.fixture(scope="module")
def cfg_train():
    return init_config(
        config_name="train.yaml",
        overrides=["numiterations=10", "debug=True", "expname=debug", "prefix=pytest"],
        config_path="../gs_config",
    )


@pytest.fixture(scope="module")
def cfg_render():
    return init_config(
        config_name="rendering.yaml",
        overrides=[
            "numiterations=10",
            "debug=True",
            "expname=debug",
            "skip_train=False",
        ],
        config_path="../gs_config",
    )


@pytest.mark.order(1)
def test_train(cfg_train):
    main_train(cfg_train)


@pytest.mark.order(2)
def test_rendering(cfg_render):
    main_render(cfg_render)


@pytest.mark.order(3)
def test_eval_dsm(cfg_render):
    main_eval_dsm(cfg_render)


@pytest.mark.order(4)
def test_tsdf(cfg_render):
    main_tsdf(cfg_render)
