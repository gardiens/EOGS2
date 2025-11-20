import torch


def pgd8(n, k=8):
    """Get greatest multiple of k <= n."""
    return (n // k) * k


from torchvision.models.optical_flow import raft_large, raft_small
import torch.nn.functional as Fnn


def ppcm8(n: int, k: int = 8) -> int:
    """Get smallest multiple of k >= n."""
    return ((n + k - 1) // k) * k


class performOpticalmatching:
    _model = None  # Class attribute for shared model

    def __init__(
        self,
        perform_cst_displacement: bool,
        mode: str = "downscale",
        device="cuda",
        model_name: str = "large",
        num_flow_updates: int = 12,
        criteria: str = "max_value_flow",
    ):
        """Perform the optical matching between two images using RAFT.

        Parameters
        ----------
        perform_cst_displacement : bool
            if True, the flow is set to the mean constant displacement
        mode : str, optional
            perform a operation to fit the required img shape, by default "downscale"
        num_flow_updates : int, optional
            number of flow updates to perform, by default 12
        """
        self.device = device
        self.perform_cst_displacement = perform_cst_displacement
        assert mode in [
            "downscale",
            "upscale",
        ], f"mode should be downscale or upscale, got {mode}"
        self.mode = mode
        self.model_name = model_name
        assert model_name in [
            "large",
            "small",
        ], f"model_name should be either large or small, got {model_name}"
        # Precompute grid templates for different sizes
        self._grid_cache = {}
        assert (
            criteria
            in [
                "max_value_flow",
                "psnr",
                "l_photom",
                "always",
            ]
        ), f"criteria should be either max_value_flow, psnr, l_photom or always, got {criteria}"
        self.criteria = criteria
        self.num_flow_updates = num_flow_updates

    def set_cst_displacement(self, flow):
        hor_shift = flow[0, 0].mean()
        vert_shift = flow[0, 1].mean()
        # if in inference mode, you can't chagne directly change the value of output_flow...
        output_flow = torch.zeros_like(flow, device=flow.device)
        output_flow[0, 0] = hor_shift
        output_flow[0, 1] = vert_shift
        return output_flow

    def _get_model(self):
        if self._model is None:
            if self.model_name == "large":
                self._model = (
                    raft_large(pretrained=True, progress=False).to(self.device).eval()
                )
            elif self.model_name == "small":
                self._model = (
                    raft_small(pretrained=True, progress=False).to(self.device).eval()
                )
        return self._model

    def normalize_img_raft(self, img):
        """Normalize the image to the range [-1, 1], suppose the range is [0,1]."""
        img = (img - 0.5) * 2
        return img

    def adjust_img_for_raft(
        self,
        normalized_img_msi_gt,
        normalized_img_msi_target,
        img_msi_gt,
        img_msi_target,
    ):
        # model has to be a multiple of 8, we apply to the batch and gt image
        # we downscale the image to the nearest multiple of 8
        if self.mode == "downscale":
            i1 = min(
                pgd8(n=normalized_img_msi_gt.shape[2]),
                pgd8(n=normalized_img_msi_target.shape[2]),
            )
            j1 = min(
                pgd8(n=normalized_img_msi_gt.shape[3]),
                pgd8(n=normalized_img_msi_target.shape[3]),
            )
            normalized_img_msi_gt = normalized_img_msi_gt[:, :, :i1, :j1]
            normalized_img_msi_target = normalized_img_msi_target[:, :, :i1, :j1]
            img_msi_gt = img_msi_gt[:, :i1, :j1]
            img_msi_target = img_msi_target[:, :i1, :j1]
            n = -1
            m = -1
        # we upscale the image to the nearest multiple of 8
        elif self.mode == "upscale":
            n, m = img_msi_gt.shape[1], img_msi_gt.shape[2]
            i1 = max(
                ppcm8(n=normalized_img_msi_gt.shape[2]),
                ppcm8(n=normalized_img_msi_target.shape[2]),
            )
            j1 = max(
                ppcm8(n=normalized_img_msi_gt.shape[3]),
                ppcm8(n=normalized_img_msi_target.shape[3]),
            )

            # We pad the image with reflect mode, we add pixel to the right and the bottom respectively.
            normalized_img_msi_gt = torch.nn.functional.pad(
                normalized_img_msi_gt,
                (
                    0,
                    j1 - normalized_img_msi_gt.shape[-1],
                    0,
                    i1 - normalized_img_msi_gt.shape[-2],
                ),
                mode="reflect",
            )
            normalized_img_msi_target = torch.nn.functional.pad(
                normalized_img_msi_target,
                (
                    0,
                    j1 - normalized_img_msi_target.shape[-1],
                    0,
                    i1 - normalized_img_msi_target.shape[-2],
                ),
                mode="reflect",
            )
        else:
            raise ValueError("mode should be either downscale or upscale")
        return (
            normalized_img_msi_gt,
            normalized_img_msi_target,
            img_msi_gt,
            img_msi_target,
            n,
            m,
        )

    def get_flow(self, img_msi_gt, img_msi_target, device="cuda"):
        """If the image has one channel, we repeat it 3 times to have 3 channels."""
        assert (
            img_msi_gt.shape[0] == img_msi_target.shape[0]
        ), " size of both images should be the same, got {} and {}".format(
            img_msi_gt.shape, img_msi_target.shape
        )
        if img_msi_gt.shape[0] == 1:
            normalized_img_msi_gt = img_msi_gt.expand(3, -1, -1)
            normalized_img_msi_target = img_msi_target.expand(3, -1, -1)
        else:
            normalized_img_msi_gt = img_msi_gt
            normalized_img_msi_target = img_msi_target

        assert (
            normalized_img_msi_gt.shape[0] == 3
        ), "Image should have 3 channels, got {}".format(normalized_img_msi_gt.shape[0])

        normalized_img_msi_gt = self.normalize_img_raft(
            normalized_img_msi_gt
        ).unsqueeze(0)
        normalized_img_msi_target = self.normalize_img_raft(
            normalized_img_msi_target
        ).unsqueeze(0)

        # model has to be a multiple of 8, we apply to the batch and gt image
        # we downscale the image to the nearest multiple of 8
        (
            normalized_img_msi_gt,
            normalized_img_msi_target,
            img_msi_gt,
            img_msi_target,
            n,
            m,
        ) = self.adjust_img_for_raft(
            normalized_img_msi_gt, normalized_img_msi_target, img_msi_gt, img_msi_target
        )

        # get the model
        model = self._get_model()
        with torch.inference_mode():
            list_of_flows = model(
                normalized_img_msi_gt.to(device),
                normalized_img_msi_target.to(device),
                num_flow_updates=self.num_flow_updates,
            )
        predicted_flows = list_of_flows[-1]
        if self.mode == "upscale":
            # we resize the flow to the original size.
            predicted_flows = predicted_flows[:, :, :n, :m]
        if self.perform_cst_displacement:
            # set the flow to the constant displacement
            predicted_flows = self.set_cst_displacement(predicted_flows)
        return predicted_flows, img_msi_gt, img_msi_target

    def _get_cached_grid(self, H: int, W: int) -> torch.Tensor:
        """Get or create cached grid for given dimensions."""
        key = (H, W)
        if key not in self._grid_cache:
            y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
            grid = torch.stack((x, y), dim=0).float().to(self.device)
            self._grid_cache[key] = grid
        return self._grid_cache[key]

    def apply_flow(self, img_msi_target, flow):
        """Warp the target image using the predicted flow."""
        if img_msi_target.dim() == 2:
            img_msi_target = img_msi_target.unsqueeze(0)  # (C, H, W)
        C, H, W = img_msi_target.shape

        img_msi_target = img_msi_target.unsqueeze(0)  # (1, C, H, W)

        # Create mesh grid
        grid = self._get_cached_grid(H, W).unsqueeze(0)  # (1, 2, H, W)

        # Add flow

        flow_grid = grid + flow

        # Normalize to [-1, 1] because grid_sample expects the range [-1,1]
        flow_grid[:, 0] = 2.0 * flow_grid[:, 0] / (W - 1) - 1.0
        flow_grid[:, 1] = 2.0 * flow_grid[:, 1] / (H - 1) - 1.0
        flow_grid = flow_grid.permute(0, 2, 3, 1)  # (1, H, W, 2)

        # Warp
        warped = Fnn.grid_sample(
            img_msi_target,
            flow_grid.detach(),  #!
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped.squeeze(0)  # (C, H, W)

    def compute_stats(self, predicted_flows, verbose: bool = False):
        if verbose:
            print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")
            print(
                f"horizontal mean = {predicted_flows[0,0].mean()}, std = {predicted_flows[0,0].std()}"
            )
            print(
                f"vertical mean = {predicted_flows[0,1].mean()}, std = {predicted_flows[0,1].std()}"
            )
        return (
            predicted_flows[0, 0].mean(),
            predicted_flows[0, 1].mean(),
            predicted_flows[0, 0].std(),
            predicted_flows[0, 1].std(),
        )

    def get_and_apply_flow(
        self, img_msi_gt, img_msi_target, device="cuda", verbose: bool = False
    ):
        """This function can work with pan images.

        In this setting, we get the flow from the gt image to the target image, then we apply the
        flow to the target image
        """
        predicted_flows, img_msi_gt, img_msi_target = self.get_flow(
            img_msi_gt, img_msi_target, device
        )
        nv_img = self.apply_flow(img_msi_target=img_msi_target, flow=predicted_flows)
        return (
            predicted_flows,
            img_msi_gt,
            nv_img,
        )


from utils.image_utils import psnr, lphotom


def perform_flow_matching(opt, warper: "performOpticalmatching", image, gt_image):
    # perform the flow matching algorithm
    predicted_flows, gt_image2, image2 = warper.get_and_apply_flow(
        img_msi_gt=gt_image,
        img_msi_target=image,
    )
    apply_flowmatch = False
    if warper.criteria == "max_value_flow":
        # if the mean value of the flow is too large, we use the original image
        if abs(predicted_flows).mean() < opt.flowmatching.max_value_flow:
            apply_flowmatch = True

    if warper.criteria == "always":
        apply_flowmatch = True

    if warper.criteria == "psnr":
        psnr_before = psnr(img1=gt_image, img2=image)
        psnr_after = psnr(img1=gt_image2, img2=image2)
        if psnr_after > psnr_before:
            apply_flowmatch = True
    if warper.criteria == "l_photom":
        Ll1 = torch.abs(gt_image - image).mean()
        Ll1_after = torch.abs(gt_image2 - image2).mean()
        lambda_dssim = 0.2
        l_photom_before = lphotom(
            image=image, gt_image=gt_image, Ll1=Ll1, lambda_dssim=lambda_dssim
        )
        l_photom_after = lphotom(
            image=image2, gt_image=gt_image2, Ll1=Ll1_after, lambda_dssim=lambda_dssim
        )
        if l_photom_after < l_photom_before:
            apply_flowmatch = True

    if apply_flowmatch:
        gt_image = gt_image2
        image = image2
    return predicted_flows, gt_image, image
