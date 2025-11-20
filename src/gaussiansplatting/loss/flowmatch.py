from .base_loss import base_Loss
from torch import Tensor


class flowmatchLoss(base_Loss):
    def __init__(self, w_L_flowmatch):
        super(flowmatchLoss, self).__init__()
        self.w_L_flowmatch = w_L_flowmatch

        pass

    def forward(self, flow) -> Tensor:
        L_flowmatch = abs(flow.mean())
        return L_flowmatch

    def get_loss_name(self) -> str:
        return "L_opacity"
