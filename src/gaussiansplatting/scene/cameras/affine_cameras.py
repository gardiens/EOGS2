import torch
from torch import nn
import numpy as np

from dataclasses import dataclass
import typing

if typing.TYPE_CHECKING:
    from utils.typing_utils import *
from copy import deepcopy


@dataclass
class SunCamera:
    world_view_transform: np.array
    full_proj_transform: np.array
    camera_center: np.array
    image_width: int
    image_height: int
    FoVx: float = 1
    FoVy: float = 1
    learn_wv_only_lastparam: bool = False

    def ECEF_to_UVA(self, xyz):
        # xyz: (..., 3)
        # We store the affine matrix as a 4x4 matrix transposed (to be compatible with CUDA code)
        At = self.world_view_transform[:3, :3]
        bt = self.world_view_transform[3, :3]
        uva = xyz @ At + bt
        return uva


class ShadowMap(nn.Module):
    def __init__(self):
        super(ShadowMap, self).__init__()

    def forward(self, shadow):
        shadow = torch.exp(0.4 * shadow.clip(max=0.0))

        return shadow


class StraightThroughHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > -0.6).float()

    @staticmethod
    def backward(ctx, grad_output):
        return nn.functional.sigmoid(grad_output + 0.6) * (
            1 - nn.functional.sigmoid(grad_output + 0.6)
        )
        # return nn.functional.hardtanh(grad_output)


def rotation_matrix_x(phi):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, torch.cos(phi), -torch.sin(phi)],
            [0, torch.sin(phi), torch.cos(phi)],
        ]
    )


def rotation_matrix_y(theta):
    return torch.tensor(
        [
            [torch.cos(theta), 0, torch.sin(theta)],
            [0, 1, 0],
            [-torch.sin(theta), 0, torch.cos(theta)],
        ]
    )


def rotation_matrix_z(alpha):
    return torch.tensor(
        [
            [torch.cos(alpha), -torch.sin(alpha), 0],
            [torch.sin(alpha), torch.cos(alpha), 0],
            [0, 0, 1],
        ]
    )


class AffineCamera(nn.Module):
    def __init__(
        self,
        caminfo: "AffineCameraInfo",
        image,
        gt_alpha_mask,
        data_device="cuda",
        is_reference_camera=False,
        image_type="msi",
        args: "ModelParams" = None,
    ):
        # camparams will be the hyperparameters of the camera, like if u wnat to se cc or world view transform
        super(AffineCamera, self).__init__()
        self.image_type = image_type
        self.image_name = caminfo.image_name
        self.is_reference_camera = is_reference_camera

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.original_image = self.compute_original_image(image=image)
        self.reference_altitude = (
            torch.from_numpy(caminfo.reference_altitude).to(self.data_device).float()
        )
        self.min_world = (
            torch.from_numpy(caminfo.min_world).to(self.data_device).float()
        )
        self.max_world = (
            torch.from_numpy(caminfo.max_world).to(self.data_device).float()
        )
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        if caminfo.width != self.image_width or caminfo.height != self.image_height:
            print(
                f"[Warning] Camera image size {self.image_width}x{self.image_height} does not match expected size {caminfo.width}x{caminfo.height}. Using the original image size and not the one provided by caminfo."
            )

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones(
                (1, self.image_height, self.image_width), device=self.data_device
            )

        # self.zfar = 100.0
        # self.znear = 0.01

        self.UV_grid = torch.meshgrid(
            torch.linspace(-1, 1, self.image_width, device=self.data_device),
            torch.linspace(-1, 1, self.image_height, device=self.data_device),
            indexing="xy",
        )

        self.centerofscene_ECEF = (
            torch.from_numpy(caminfo.centerofscene_ECEF).to(self.data_device).float()
        )
        affine_coef = torch.from_numpy(caminfo.affine_coef)
        affine_inter = torch.from_numpy(caminfo.affine_inter)

        self.affine = torch.eye(4, 4)
        self.affine[:3, :3] = affine_coef
        self.affine[:3, -1] = affine_inter

        self.affine = (
            self.affine.to(self.data_device).float().T
        )  # ! BEWARE OF THE TRANSPOSE !

        self.Ainv = torch.inverse(self.affine[:3, :3].T)

        if not caminfo.load_sun:
            if not (
                caminfo.sun_affine_coef is None or caminfo.sun_affine_inter is None
            ):
                print(" sun affine coefficient provided, but not loaded")

            self.sun_affine = None
        else:
            self.sun_affine = torch.eye(4, 4)
            self.sun_affine[:3, :3] = torch.from_numpy(caminfo.sun_affine_coef)
            self.sun_affine[:3, -1] = torch.from_numpy(caminfo.sun_affine_inter)

            self.sun_affine = self.sun_affine.to(self.data_device).float().T

            self._camera_to_sun = (
                torch.from_numpy(caminfo.camera_to_sun).to(self.data_device).float()
            )

        self.altitude_bounds = (
            torch.from_numpy(caminfo.altitude_bounds).to(self.data_device).float()
        )

        self.FoVx = 1
        self.FoVy = 1

        self.world_view_transform = self.affine
        self.full_proj_transform = self.affine  # not used I think
        self.camera_center = torch.zeros(3, device=self.data_device)
        # self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]
        if (
            caminfo.learn_wv_transform
        ):  # If we want to learn the world to view transform
            # We learn the world to view transform
            # It doens't work yet
            if args.camera_params.learn_wv_only_lastparam:
                self.learn_wv_only_lastparam = True
                self.last_row = nn.Parameter(torch.zeros(4, device=self.data_device))
                print("Learning only the last parameter of the world to view transform")
                self.last_row.requires_grad = False
            else:
                self.learn_wv_only_lastparam = False

                self.world_view_transform = nn.Parameter(
                    self.affine.clone().to(self.data_device)
                )
                self.full_proj_transform = nn.Parameter(
                    self.affine.clone().to(self.data_device)
                )
            # freeze the camera for now
            self.world_view_transform.requires_grad = False
            self.learn_wv_transform = True
        else:
            self.learn_wv_transform = False
            self.learn_wv_only_lastparam = False
        # Optimize per-camera color correction
        if caminfo.use_cc:
            self.use_exposure = False
            self.use_cc = True
            self.color_correction = nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=1, bias=True
            ).to(self.data_device)
            with torch.no_grad():
                self.color_correction.weight = nn.Parameter(
                    torch.eye(3, device=self.data_device).reshape(3, 3, 1, 1)
                )
                self.color_correction.bias = nn.Parameter(
                    torch.zeros(3, device=self.data_device)
                )

            self.inshadow_color_correction = nn.Parameter(
                torch.zeros(3, device=self.data_device).reshape(3, 1, 1) + 0.05
            )
            # # self.inshadow_color_correction = nn.Conv2d(3, 3, 1, bias=False).to(self.data_device)
            # with torch.no_grad():
            #     # self.inshadow_color_correction.weight = nn.Parameter(torch.eye(3, device=self.data_device).reshape(3,3,1,1) * 0.05)
            #     # self.inshadow_color_correction.bias = nn.Parameter(torch.zeros(3, device=self.data_device)+0.2)
            # # self.inshadow_color_correction = nn.Parameter(torch.tensor([0.66], device=self.data_device))

        else:
            self.use_cc = False
            self.use_exposure = False

            if args.camera_params.use_exposure:
                self.use_exposure = True
                exposure = torch.eye(3, 4, device="cuda")[None]
                self.exposure = nn.Parameter(exposure.requires_grad_(True))
            else:
                print("are you sure you don't use exposure and color correction?")
                print(" we will use a base color correction anyway")
                self.color_correction = nn.Conv2d(3, 3, 1, bias=True).to(
                    self.data_device
                )
                with torch.no_grad():
                    self.color_correction.weight = nn.Parameter(
                        torch.eye(3, device=self.data_device).reshape(3, 3, 1, 1)
                    )
                    self.color_correction.bias = nn.Parameter(
                        torch.zeros(3, device=self.data_device)
                    )

                self.inshadow_color_correction = nn.Parameter(
                    torch.zeros(3, device=self.data_device).reshape(3, 1, 1) + 0.05
                )
                self.use_exposure = False

        self.inshadow_color_correction = nn.Parameter(
            torch.zeros(3, device=self.data_device).reshape(3, 1, 1) + 0.05
        )
        self.use_shadow = caminfo.use_shadow
        if not self.use_shadow:
            print(
                "WARNING: Shadow mapping is disabled for this camera. This may lead to unrealistic lighting effects."
            )
        # Shadow color correction
        self.shadow_map = ShadowMap().to(self.data_device)
        # add Transient material
        if args.transient_params.use_transient:
            self.use_transient = True
            self.transient_mask = nn.Parameter(
                torch.full(
                    (self.image_height, self.image_width),
                    args.transient_params.init_value,  # Initial value
                    device=self.data_device,
                    requires_grad=True,
                )
            )
            self.transient_canvas = nn.Parameter(self.original_image.clone())
        else:
            self.use_transient = False
        return

    def compute_original_image(self, image):
        """Get the original image: The output shape should be [C,H,W]"""
        return image.to(self.data_device)
        return image.clamp(0.0, 1.0).to(self.data_device)

    # def activate_learning_world_view_transform(self):
    #     assert self.learn_wv_transform, "Camera is not set to learn world view transform"
    #     self.world_view_transform.requires_grad = True
    def render_pipeline(self, raw_render, sun_altitude_diff=None):
        # raw_render : (3,H,W)
        # sun_altitude_diff : (H,W) or None

        render = raw_render.unsqueeze(0)

        # Step 2: Apply color correction
        if self.use_cc:  # not self.is_reference_camera:
            cc = self.color_correction(render)
        elif self.use_exposure:
            exposure = self.exposure
            B, C, H, W = render.shape

            # flatten pixels -> (B, 3, H*W)
            render_flat = render.view(B, C, -1)

            # affine transform: (B, 3, 3) @ (B, 3, N) + bias
            cc = torch.bmm(exposure[:, :, :3], render_flat) + exposure[:, :, 3:4]

            # reshape back to image
            cc = cc.view(B, C, H, W)

        else:
            cc = render

        # Step 1: Apply shades
        if self.use_shadow and sun_altitude_diff is not None:
            shadow = self.shadow_map(
                sun_altitude_diff
            )  # Now is between 0 and 1, dimensions are (H,W)
            shaded = shadow * cc + (1 - shadow) * self.inshadow_color_correction * cc
        else:
            shadow = None
            shaded = cc

        final = shaded
        # paint transient on top.
        # if self.use_transient:
        #     mask = self.transient_mask.clip(0, 1)
        #     final = final * (1 - mask) + self.transient_canvas * mask
        return {
            "shadowmap": shadow,
            "shaded": shaded.squeeze(0),
            "cc": cc.squeeze(0),
            "final": final.squeeze(0),
        }

    def get_sun_camera(self, f=2) -> "SunCamera":
        # The sun camera rendering is done with larger footprint (and resolution)
        if self.sun_affine is None:
            raise ValueError(
                f"Sun affine transformation is not provided for this camera. The camera is {self.image_name}. "
            )
        scalingmat = torch.eye(4, device="cuda")
        scalingmat[0, 0] = 1 / f
        scalingmat[1, 1] = 1 / f

        cam2virt = scalingmat[:3, :3] @ self._camera_to_sun

        return SunCamera(
            world_view_transform=self.sun_affine @ scalingmat,
            full_proj_transform=self.sun_affine @ scalingmat,
            camera_center=self.camera_center,
            image_width=self.image_width * f,
            image_height=self.image_height * f,
            FoVx=self.FoVx,
            FoVy=self.FoVy,
        ), cam2virt

    def get_nadir_camera(self, f=1):
        A = self.affine[:3, :3].T
        b = self.affine[3, :3]

        d = torch.zeros(3, device="cuda")
        d[-1] = 1
        q = A @ d
        q = q / q[-1]
        myM = torch.eye(3, device=self.data_device)
        myM[:2, 2] = -q[:2]

        new_A = myM @ A
        new_b = (
            torch.eye(3, device=myM.device) - myM
        ) @ A @ self.centerofscene_ECEF + b

        new_affine = torch.eye(4, 4)
        new_affine[:3, :3] = new_A
        new_affine[:3, -1] = new_b
        new_affine = new_affine.to(self.data_device).float().T

        return SunCamera(
            world_view_transform=new_affine,
            full_proj_transform=new_affine,
            camera_center=self.camera_center,
            image_width=self.image_width,
            image_height=self.image_height,
            FoVx=self.FoVx,
            FoVy=self.FoVy,
        ), myM

    def sample_random_camera(self, virtual_camera_extent):
        A = self.affine[:3, :3].T
        b = self.affine[3, :3]
        myM = torch.eye(3, device=self.data_device)
        myM[:2, 2] = (
            myM[:2, 2]
            + torch.randn(2, device=self.data_device).clip(-1, 1)
            * virtual_camera_extent
        )
        new_A = myM @ A
        new_b = (
            torch.eye(3, device=myM.device) - myM
        ) @ A @ self.centerofscene_ECEF + b

        new_affine = torch.eye(4, 4)
        new_affine[:3, :3] = new_A
        new_affine[:3, -1] = new_b
        new_affine = new_affine.to(self.data_device).float().T

        return SunCamera(
            world_view_transform=new_affine,
            full_proj_transform=new_affine,
            camera_center=self.camera_center,
            image_width=self.image_width,
            image_height=self.image_height,
            FoVx=self.FoVx,
            FoVy=self.FoVy,
        ), myM

    def ECEF_to_UVA(self, xyz):
        # xyz: (..., 3)
        # We store the affine matrix as a 4x4 matrix transposed (to be compatible with CUDA code)
        At = self.affine[:3, :3]
        bt = self.affine[3, :3]
        uva = xyz @ At + bt
        return uva

    def UVA_to_ECEF(self, uva):
        # uva: (..., 3)
        # We store the affine matrix as a 4x4 matrix transposed (to be compatible with CUDA code)
        b = self.affine[3, :3].double()
        xyz = torch.einsum(
            "...ij,...j->...i", self.Ainv.double(), uva.double() - b.double()
        )
        return xyz

    def __repr__(self):
        inter_str = super().__repr__()
        inter_str = inter_str + f"\n Camera_name: {self.image_name}"
        return super().__repr__()

    def clone(self):
        # didn't try myself but should work
        return deepcopy(self)
