# This script is the Up module of the UNet. It is used to upsample the input by a factor of 2.
# It consists of a DoubleConv layer followed by a ConvTranspose2d layer.
from .double_conv import DoubleConv
import torch.nn as nn
import torch


class Up(nn.Module):
    """Up module of the UNet

    Parameters:
    -----------
        in_channel : int
            Number of input channels.
        out_channel : int
            Number of output channels.
        padding : int
            Padding to be added to the input.
        norm_type : str
            Type of normalization to be used.
        s : int
            Size of the input.

    Return:
    -----------
        x : torch.Tensor
            Output tensor.

    """

    def __init__(
        self, in_channel, out_channel, padding=None):
        super(Up, self).__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_channel, out_channel, kernel_size=2, stride=2
        )
        self.double_conv = DoubleConv(in_channel, out_channel)
        self.padding = padding

    def forward(self, x1, x2):
        # Upsample the input
        x1 = self.up_conv(x1)

        # Calculate the padding to be added to the input if needed
        if self.padding is not None:
            x1 = self.padding(x1)

        # Concatenate the input and the output of a skip connection
        x = torch.cat([x2, x1], dim=1)

        # Pass through the DoubleConv layer
        x = self.double_conv(x)

        return x
