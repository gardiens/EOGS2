from torch import Tensor
from torch import nn


class base_Loss(nn.Module):
    def __init__(self, weight=-1):
        self.weight = weight
        super(base_Loss, self).__init__()
        return

    def forward(self, gaussians) -> Tensor:
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_loss_name(self) -> str:
        return self.__class__.__name__

    def log_loss(
        self, tb_writer: "SummaryWriter", loss_value: Tensor, iteration: int
    ) -> None:
        tb_writer.add_scalar(f"loss/{self.get_loss_name()}", loss_value, iteration)
