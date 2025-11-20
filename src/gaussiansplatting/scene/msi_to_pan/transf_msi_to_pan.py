import torch
from torch import nn


class base_msi_to_pan(nn.Module):
    def __init__(
        self,
    ):
        """Base Thibault implementation of the fixed msi to pan conversion."""
        super(base_msi_to_pan, self).__init__()
        self.register_buffer(
            "pan_params",
            torch.tensor([0.438469, 1.1331377, -0.6794343, 1.0, 0.0016913427]),
        )
        # TODO make something more generic, for now use the one from WV3
        # see 10.1109/IGARSS53475.2024.10641439
        # IDEA define these into the camera info?

    def forward(self, x):
        shaded = self.pan_params[3] * (
            torch.sum(self.pan_params[None, :3, None, None] * x, dim=-3, keepdims=True)
            + self.pan_params[4]
        )
        return shaded


class average_msitopan(nn.Module):
    def __init__(
        self,
    ):
        """Base Thibault implementation of the fixed msi to pan conversion."""
        super(average_msitopan, self).__init__()

    def forward(self, x):
        # average on the first axis
        shaded = torch.mean(x, dim=1, keepdims=True)
        return shaded


class msi_to_pan_identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """Identity function for msi to pan conversion.

        Usefull if needed for pansharpening
        """
        return x


class only_one_channel(nn.Module):
    # we keep only one channel, The first one for instance
    def __init__(self, num_channel: int = 0):
        super(only_one_channel, self).__init__()
        self.num_channel = num_channel

    def forward(self, x):
        return x[:, self.num_channel : self.num_channel + 1, :, :]


class learnable_base_msi_to_pan(nn.Module):
    def __init__(
        self,
    ):
        """Base Thibault implementation of the fixed msi to pan conversion."""
        super(learnable_base_msi_to_pan, self).__init__()
        self.register_parameter(
            "pan_params",
            nn.Parameter(
                torch.tensor([0.438469, 1.1331377, -0.6794343, 1.0, 0.0016913427]),
                requires_grad=False,
            ),
        )
        for param in self.pan_params:
            param.requires_grad = False
        self.learn_conved2d = False  #! not used but to be consistent with other code?

    def forward(self, x):
        shaded = self.pan_params[3] * (
            torch.sum(self.pan_params[None, :3, None, None] * x, dim=-3, keepdims=True)
            + self.pan_params[4]
        )
        return shaded


class MSI_TO_PAN(nn.Module):
    def __init__(
        self,
        msi_channels,
        pan_channel,
        kernel_size=1,
        remove_sigm: bool = False,
        use_avgpool: bool = False,
        init_value: bool = False,
    ):
        super(MSI_TO_PAN, self).__init__()
        self.msi_channels = msi_channels
        self.pan_channel = pan_channel
        self.linear = nn.Conv2d(
            msi_channels, pan_channel, kernel_size=kernel_size, padding="same"
        )
        if use_avgpool:
            self.linear = nn.AvgPool2d(kernel_size=kernel_size, ceil_mode=True)
            print("WE USE A AVG POOL 2D ")
        self.activation = nn.Sigmoid()
        self.remove_sigm = remove_sigm
        if init_value:
            self.init_value()
        # bias is already included

    def init_value(self):
        with torch.no_grad():
            # Set weights: shape (out_channels, in_channels, 1, 1)
            print("we init value")
            weights = torch.tensor([[[[0.438469]], [[1.1331377]], [[-0.6794343]]]])
            bias = torch.tensor([0.0016913427])

            self.linear.weight.copy_(weights)
            self.linear.bias.copy_(bias)

    def forward(self, x):
        """Convert a MSI image to PAN image assume x of shape C H W."""

        x = self.linear(x)
        if self.remove_sigm:
            x = x
        else:
            x = self.activation(x)

        return x


class msi_to_pan_fixedandtranslate(nn.Module):
    # fixed version?
    def __init__(
        self,
        msi_channels,
        pan_channel,
        kernel_size=1,
        init_value: bool = False,
    ):
        super(msi_to_pan_fixedandtranslate, self).__init__()
        self.msi_channels = msi_channels
        self.pan_channel = pan_channel
        self.linear = nn.Conv2d(
            msi_channels, pan_channel, kernel_size=kernel_size, padding="same"
        )
        self.activation = nn.Sigmoid()
        if init_value:
            self.init_value()
        # bias is already included
        # Fixed weights and bias as a buffer (non-learnable)
        weights = torch.tensor([[[[0.438469]], [[1.1331377]], [[-0.6794343]]]])
        bias = torch.tensor([0.0016913427])
        self.register_buffer("fixed_weights", weights)
        self.register_buffer("fixed_bias", bias)
        self.learn_conv2d = False  # If true, we learn the conv2d weights

    def init_value(self):
        pass

    def forward(self, x):
        """Convert a MSI image to PAN image assume x of shape C H W."""

        # Compute y using fixed weights (no gradient)
        with torch.no_grad():
            y = torch.sum(
                self.fixed_weights * x, dim=1, keepdim=True
            ) + self.fixed_bias.view(1, 1, 1, 1)

        if self.learn_conv2d:
            x = self.linear(x)

            x = x + y
        else:
            x = y
        return x


# width = 528
# height = 528
# MSI_chan = 3
# PAN_chan = 1
# msi_to_pan = MSI_TO_PAN2(MSI_chan, PAN_chan,kernel_size=1).to("cuda")
# x = torch.rand(MSI_chan, height, width, device="cuda")  # Add batch dimension


def load_msi_to_pan(msi_pan_cfg: "cfg") -> "MSI_TO_PAN":
    if msi_pan_cfg.name == "only_one_channel":
        msi_pan = only_one_channel(num_channel=0)  #!
    elif msi_pan_cfg.name == "base":
        msi_pan = MSI_TO_PAN(
            msi_channels=msi_pan_cfg.msi_channels,
            pan_channel=msi_pan_cfg.pan_channels,
            kernel_size=msi_pan_cfg.kernel_size,
            remove_sigm=msi_pan_cfg.remove_sigm,
            init_value=msi_pan_cfg.init_value,
            use_avgpool=msi_pan_cfg.use_avgpool,
        )
    elif msi_pan_cfg.name == "average":
        msi_pan = average_msitopan()
    elif msi_pan_cfg.name == "fixed":
        # print("we are takinga fixed MSI TO PAN ?")
        msi_pan = base_msi_to_pan()
    elif msi_pan_cfg.name == "fixedandtranslate":
        msi_pan = msi_to_pan_fixedandtranslate(
            msi_channels=msi_pan_cfg.msi_channels,
            pan_channel=msi_pan_cfg.pan_channels,
            kernel_size=msi_pan_cfg.kernel_size,
            init_value=msi_pan_cfg.init_value,
        )
    elif msi_pan_cfg.name == "learnable_fixed":
        msi_pan = learnable_base_msi_to_pan()
    elif msi_pan_cfg.name == "identity":
        msi_pan = msi_to_pan_identity()
    else:
        raise ValueError(
            f"Unknown MSI to PAN conversion type: {msi_pan_cfg.name}. "
            "Available options are 'base', 'fixed', or 'fixedandtranslate'."
        )
    return msi_pan


if __name__ == "__main__":
    width = 528
    height = 528
    MSI_chan = 3
    PAN_chan = 1
    msi_to_pan = MSI_TO_PAN(MSI_chan, PAN_chan).to("cuda")
    x = torch.rand(MSI_chan, height, width, device="cuda")  # Add batch dimension
    print("x shape", x.shape)
    print("output shape", msi_to_pan(x).shape)
