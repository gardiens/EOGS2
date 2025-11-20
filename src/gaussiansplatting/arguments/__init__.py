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

from argparse import ArgumentParser, Namespace
import sys
import os

from omegaconf import OmegaConf, DictConfig
import hydra


class GroupParams:
    pass


class ParamGroupHydra(GroupParams):
    def __init__(self, cfg):
        for key in cfg:
            # check that the attr is already defined
            if not hasattr(self, key) and not hasattr(self, "_" + key):
                raise AttributeError(
                    f" the Attribute {key} initiated in the cfg is  not defined in the python config {self.__class__.__name__}, the cfg is {cfg}"
                )
            setattr(self, key, cfg[key])


class flowmatchingParams(ParamGroupHydra):
    def __init__(self, cfg):
        self.apply_flowmatching = False  # apply flow matching
        self.flowmatch_msi = True
        self.flowmatch_pan = True
        self.max_value_flow = 2  # maximum value of the flow
        self.perform_cst_displacement = (
            False  # if true, the flow is set to the mean displacement
        )
        self.mode = "downscale"  # mode to fit the image to the model requirements,
        self.model_name = "large"  # raft_large or raft_small
        self.iterend_flowmatching = 9999999
        self.num_flow_updates = 12
        self.criteria = "max_value_flow"  # max_value_flow or photometric
        super().__init__(cfg)


class early_stopping(ParamGroupHydra):
    def __init__(self, cfg):
        self.patience = 200
        self.operator = "max"  # 'min' or 'max'
        self.metric_name = "psnr"
        self.use_early_stopping = True
        super().__init__(cfg)


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t is bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t is bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args) -> "GroupParams":
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ClearmlParams(ParamGroupHydra):
    def __init__(self, cfg=None):
        self.task_name = ""
        self.project_name = "EOGS"
        self.resume_clearml = False
        self.tags = []
        super().__init__(cfg)


class densificationParams(ParamGroupHydra):
    def __init__(self, cfg):
        self.only_prune: bool = True
        self.densify_from_iter: int = 500
        self.densification_interval: int = 100
        self.densify_grad_threshold: float = 2e-6
        super().__init__(cfg)


class CameraParams(ParamGroupHydra):
    def __init__(self, cfg):
        self.use_cc = False  # adjust the color correction
        self.learn_wv_transform: bool = False  # learn the world to view transform
        self.use_only_rot = False  # use only rotation in the world to view transform
        self.pan_images = ""  # folder to panchromatic images
        self.use_shadow = True  # use shadow mapping
        self.learn_wv_only_lastparam: bool = (
            False  # learn only the last parameter of the world to view transform
        )
        self.use_exposure: bool = False  # learn exposure per camera

        super().__init__(cfg)

    def extract(self, args):
        tmp = super().extract(args)
        if not tmp.use_cc:
            print("WARNING We are not using  color correction, are you sure about it?")

        return tmp


class MSI_TO_PANParams(ParamGroupHydra):
    def __init__(self, cfg):
        self.kernel_size = 3  # kernel size for the MSI to PAN conversion
        self.msi_channels = 3  # number of MSI channels
        self.pan_channels = 1  # number of PAN channels
        self.remove_sigm = False  # remove the sigmoid activation
        self.init_value = False
        self.use_avgpool = False  # use average pooling instead of linear layer
        self.name = "base_msi"
        super().__init__(cfg)


class loggingGS(ParamGroupHydra):
    def __init__(self, cfg):
        self.tb_log_interval = 10  # log interval for tensorboard
        self.pan_log_interval = 1600  # log interval for pancrhomatic images
        self.dsm = DSMParams(cfg.get("dsm", None))  # DSM parameters
        self.big_testing_iterations = []
        super().__init__(cfg)


class DSMParams(ParamGroupHydra):
    def __init__(self, cfg):
        self.gt_dir = ""

        self.aoi_id = ""
        self.enable_vis_mask = False
        self.filter_tree = False
        self.use_another_computer_filter_tree = False
        super().__init__(cfg)


class initialize_pcdParams(ParamGroupHydra):
    def __init__(self, cfg):
        self.input_ply_name = None  # if not null,we load the ply
        super().__init__(cfg)


class ModelParams(ParamGroupHydra):
    def __init__(self, cfg):
        self.sh_degree = 3
        self._source_path = ""  # affine model path
        self.images_pan_path = ""  # path to the panchromatic images
        self.images_msi_path = ""  # path to the multispectral images
        self.load_pan = False  # load panchromatic images
        self.load_msi = False  # load multispectral images
        self._model_path = ""  # output model path
        self.scene_name = ""  # scene name
        # self._images = "images"
        self.need_rescale = (
            False  # If True, images will be rescaled by 255 ( deal to int to float)
        )
        self.rescaler_name = "identity"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.target_density = 0.13  # target density for the point cloud
        self.scale_factor_z = 1
        self.camera_params = CameraParams(cfg.get("camera_params", None))
        self.msi_to_pan = MSI_TO_PANParams(cfg.get("msi_to_pan", None))
        self.share_color_correction = False
        self.share_worldview_transform = False
        self.transient_params = TransientParams(cfg.get("transient_params", None))
        self.weird_pan_setup: bool = False
        self.opacity_init_value: float = 0.01
        self.repeat_gt: bool = False  # repeat the ground truth images if needed
        self.initialize_pcd = initialize_pcdParams(cfg.get("initialize_pcd", None))
        self.train_to_test_cc_converter = None
        super().__init__(cfg)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        g.camera_params = self.camera_params.extract(args)
        g.msi_to_pan = self.msi_to_pan.extract(args)
        g.transient_params = self.transient_params.extract(args)
        g.initialize_pcd = self.initialize_pcd.extract(args)
        return g


class TransientParams(ParamGroupHydra):
    def __init__(self, cfg):
        self.init_value = 0.01
        self.use_transient = False
        super().__init__(cfg)


class PipelineParams(ParamGroupHydra):
    def __init__(self, cfg=None):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        self.require_radii = (
            False  # if True, we output in the rendering of the gauissian
        )

        super().__init__(cfg)

    def extract(self, args):
        g = super().extract(args)
        # g.msi_to_pan = self.msi_to_pan.extract(args)
        return g


class pansharpParams(ParamGroupHydra):
    def __init__(self, cfg):
        self.method = "None"  # pansharpening method


class OptimizationParams(ParamGroupHydra):
    def __init__(self, cfg):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.opacity_reset_interval = 3000
        self.iterend_opacity_reset_interval = 99999999
        self.densify_until_iter = 15_000
        self.random_background = True  # False # TODO: Maybe should be True?
        self.copy_background_firschan = False
        self.optimizer_type = "default"
        self.only_prune = True
        #####################################
        # EOGS specific parameters
        #####################################
        # load or not the pan or msi images
        self.load_pan = True  # load panchromatic images
        self.load_msi = True  # load multispectral images

        # action starting iterations
        self.iterstart_shadowmapping = 1000
        self.color_reset_iterations = 9999999999

        self.itr_apply_flowmatching_to_affine = (
            -1
        )  # Iteration to apply the flowmatching to the affine transformation
        # loss terms starting iterations
        self.iterstart_L_opacity = -1  # TODO: start at 1000
        self.iterend_L_opacity = 9999999999
        self.iterstart_L_sun_resample = 9999999999
        self.iterstart_L_new_resample = 1000
        self.iterstart_L_TV_altitude = 9999999999
        self.iterstart_L_erank = -1
        self.iterstart_L_nll = 9999999999
        self.iterstart_L_accumulated_opacity = 9999999999
        self.iterstart_Lpan = 100
        self.iterstart_Lgradient_pan = 100
        self.apply_pansharp = False
        self.iterstart_L_photometric = -1
        # learning weight
        self.iterstart_learn_wv_transform = 1
        self.iterstart_learn_msitopan_params = 5000
        self.iterstart_L_opacity_radii = 999999
        self.iterend_L_opacity_radii = 9999999
        self.iterstart_L_flowmatch = 99999999
        self.iterend_L_flowmatch = 9999999
        self.freeze_start_msitopan_params = (
            True  # freeze the msi to pan parameters at the beginning
        )
        # loss terms weights
        self.w_L_photometric = 1.0
        self.w_L_opacity = 0.10
        self.w_L_opacity_radii = 0
        self.w_L_sun_altitude_resample = 0.01
        self.w_L_sun_rgb_resample = 0.10
        self.w_L_new_altitude_resample = 0.01
        self.w_L_new_rgb_resample = 0.10  # maybe 0.3 is better
        self.w_L_TV_altitude = 0.10
        self.w_L_erank = 0.0
        self.w_L_nll = 0.10
        self.w_L_translucentshadows = 1e-2
        self.w_L_accumulated_opacity = 0.0
        self.w_Lpan = 0.1
        self.w_Lgradient_pan = 0.1
        self.w_L_flowmatch = 0.0
        self.w_L_pansharp = 0.1

        # lr terms
        self.camera_lr = 1e-2
        self.msi_pan_lr = 1e-4

        # other hyperparameters
        self.virtual_camera_extent = 0.01
        # self.shadowmap_resample:bool= False
        self.normalize_colors_before_saving = True  # normalize colors before saving
        self.pansharp_cfg = pansharpParams(cfg=cfg.get("pansharp_cfg", None))

        self.min_opacity = -6.0  # minimum opacity for pruning

        self.flowmatching = flowmatchingParams(cfg=cfg.get("flowmatching", None))
        self.iterstart_flowmatching = 1500  # iterations to start flow matching

        self.random_camera = randomcameraParams(cfg=cfg.get("random_camera", None))
        self.early_stopping = early_stopping(cfg=cfg.get("early_stopping", None))
        self.densification_strategy = densificationParams(
            cfg=cfg.get("densification_strategy", None)
        )
        super().__init__(cfg)

    def extract(self, args):
        g = super().extract(args)
        g.pansharp_cfg = self.pansharp_cfg.extract(args)
        g.flowmatching = self.flowmatching.extract(args)
        g.random_camera = self.random_camera.extract(args)
        g.early_stopping = self.early_stopping.extract(args)
        return g


class randomcameraParams(ParamGroupHydra):
    def __init__(self, cfg=None):
        self.randomcamera_render_type = "rawrender"  # rawrender or rawrender_wshadow
        self.use_gt = False
        super().__init__(cfg)

    def extract(self, args):
        g = super().extract(args)
        g.pansharp_cfg = self.pansharp_cfg.extract(args)
        g.flowmatching = self.flowmatching.extract(args)
        return g


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args.yaml")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)
    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict)


def get_combined_cfg(cfg: "Dictconfig") -> " Dictconfig":
    try:
        cfgfilepath = os.path.join(cfg.model.model_path, "cfg_args.yaml")
        print("Looking for config file in", cfgfilepath)
        cfgfile = OmegaConf.load(cfgfilepath)
    except TypeError:
        print("Config file not found at")
        pass
    cfg.model = OmegaConf.unsafe_merge(cfgfile, cfg.model)
    return cfg
