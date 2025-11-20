#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import pyrootutils

root = str(
    pyrootutils.setup_root(
        search_from=__file__, indicator=".project-root", pythonpath=True, dotenv=True
    )
)
from train_pan import main as main_train
from render_pan import main as main_render
from eval.eval_dsm import main_hydra_dsm as main_eval_dsm
from tsdf import main_hydra_tsdf as main_tsdf

if __name__ == "__main__":
    # run the training
    main_train()
    # run the rendering
    main_render()
    # run the evaluation
    main_eval_dsm()
    # run the tsdf extraction
    main_tsdf()
