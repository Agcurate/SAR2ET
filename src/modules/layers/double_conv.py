# This script is the DoubleConv module of the UNet.
import torch.nn as nn


class DoubleConv(nn.Module):
    """DoubleConv module of the UNet

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
        super(DoubleConv, self).__init__()

        # Create the DoubleConv layer
        self.network = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        # Pass through the DoubleConv layer
        x = self.network(x)

        return x
