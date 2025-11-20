from torch import Tensor


from .base_loss import base_Loss


class OpacityLoss(base_Loss):
    def __init__(self, w_L_opacity, init_number_of_gaussians):
        super(OpacityLoss, self).__init__()
        self.w_L_opacity = w_L_opacity
        self.init_number_of_gaussians = init_number_of_gaussians
        pass

    def forward(self, gaussians) -> Tensor:
        opacity = gaussians.get_opacity.squeeze()
        L_opacity = opacity.sum() / self.init_number_of_gaussians
        return L_opacity

    def get_loss_name(self) -> str:
        return "L_opacity"


class radiiOpacityLoss(base_Loss):
    def __init__(self, w_L_opacity, init_number_of_gaussians):
        super(radiiOpacityLoss, self).__init__()
        self.w_L_opacity = w_L_opacity
        self.init_number_of_gaussians = init_number_of_gaussians
        pass

    def forward(self, gaussians, radii) -> Tensor:
        visible = radii > 0

        opacity = gaussians.get_opacity.squeeze()
        L_opacity_radii = opacity[visible].sum() / self.init_number_of_gaussians
        return L_opacity_radii


class AccumulatedOpacity(base_Loss):
    def __init__(self, w_L_accumulated_opacity):
        super(AccumulatedOpacity, self).__init__()
        self.w_L_accumulated_opacity = w_L_accumulated_opacity
        pass

    def forward(self, accumulated_opacity_render) -> Tensor:
        return (1.0 - accumulated_opacity_render).mean()
