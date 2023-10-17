# This script is the Down module of the UNet.
# It is used to downsample the input by a factor of 2.
from .double_conv import DoubleConv
import torch.nn as nn


class Down(nn.Module):
    """Down module of the UNet

    Parameters:
    -----------
        in_channel : int
            Number of input channels.
        out_channel : int
            Number of output channels.
        norm_type : str
            Type of normalization to be used.
        s : int
            Size of the input.

    Return:
    -----------
        x : torch.Tensor
            Output tensor.

    """

    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(in_channel, out_channel)

    def forward(self, x):
        # Pass through the MaxPool layer
        x = self.pool(x)

        # Pass through the DoubleConv layer
        x = self.double_conv(x)
        
        return x
