from torch import nn
import typing

if typing.TYPE_CHECKING:
    from utils.typing_utils import *


class MSAffineCamera(nn.Module):
    def __init__(
        self,
        Affinecamera_dict: "dict",
        is_reference_camera: bool,
        image_name,
        data_device,
        args: "ModelParams" = None,
    ):
        assert (
            "msi" in Affinecamera_dict
        ), "MSI camera info not found in the metadata, are you sure you wanna use MSI camera info?"
        super(MSAffineCamera, self).__init__()
        self.dictCamera = nn.ModuleDict(Affinecamera_dict)
        if args.share_color_correction:
            if args.camera_params.use_cc:
                self.share_color_correction(dictCamera=self.dictCamera)
            if args.camera_params.use_exposure:
                self.share_exposure(dictCamera=self.dictCamera)
        if args.share_worldview_transform:
            print("we share the wv transform")
            self.share_worldview_transform(dictCamera=self.dictCamera)
        if args.transient_params.use_transient:
            print("we use transient material for MSAffineCamera")

        # this is a affinecameradict
        self.image_name = Affinecamera_dict["msi"].image_name
        self.set_is_reference_camera(is_reference_camera=is_reference_camera)
        self.image_name = image_name
        self.data_device = data_device
        # if "pan" in Affinecamera_dict:
        #     pan_channel= Affinecamera_dict["pan"].image.shape[0]
        #     msi_channel= Affinecamera_dict["msi"].image.shape[0]
        #     self.msi_to_pan = MSI_TO_PAN(msi_channel, pan_channel).to(self.data_device)

    def set_is_reference_camera(self, is_reference_camera: bool):
        self.is_reference_camera = is_reference_camera
        for key in self.dictCamera:
            self.dictCamera[key].is_reference_camera = is_reference_camera

    def share_color_correction(self, dictCamera: "dict"):
        ref_key = "msi"
        if ref_key not in dictCamera:
            print(
                "WARNING: we do not have msi camera, we will not share the color correction"
            )
            return
        ref_cc = dictCamera[ref_key].color_correction
        for key in dictCamera:
            if dictCamera[key].color_correction is None:
                print(
                    f"WARNING: camera {key} does not have color correction, we will not share the color correction"
                )
                continue
            if key != ref_key:
                if dictCamera[key].color_correction.in_channels != ref_cc.in_channels:
                    print(
                        f"WARNING: camera {key} has different number of channels in color correction, we will still share but are you sure about that?"
                    )
                dictCamera[key].color_correction = ref_cc

    def share_exposure(self, dictCamera: "dict"):
        ref_key = "msi"
        if ref_key not in dictCamera:
            print(
                "WARNING: we do not have msi camera, we will not share the color correction"
            )
            return
        ref_cc = dictCamera[ref_key].exposure
        for key in dictCamera:
            if dictCamera[key].exposure is None:
                print(
                    f"WARNING: camera {key} does not have exposure, we will not share the color correction"
                )
                continue
            if key != ref_key:
                dictCamera[key].exposure = ref_cc

    def share_worldview_transform(self, dictCamera: "dict"):
        ref_key = "msi"
        if ref_key not in dictCamera:
            print(
                "WARNING: we do not have msi camera, we will not share the world view transform"
            )
            return
        ref_wv = dictCamera[ref_key].world_view_transform
        for key in dictCamera:
            if dictCamera[key].world_view_transform is None:
                print(
                    f"WARNING: camera {key} does not have world view transform, we will not share the world view transform"
                )
                continue
            if key != ref_key:
                dictCamera[key].world_view_transform = ref_wv

    def get_msi_cameras(self):
        return self.dictCamera["msi"]

    def get_pan_cameras(self):
        if "pan" in self.dictCamera:
            return self.dictCamera["pan"]
        else:
            return None
