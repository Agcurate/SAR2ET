# Description: This file contains the UNet class which consists of a 
# slightly customized UNet architecture.
from .layers import DoubleConv, Down, Up
import torch.nn as nn


class UNet(nn.Module):
    """
    A slightly customized UNet architecture.
    Please see https://arxiv.org/abs/1505.0459 for more detail.
    """

    def __init__(self, d_in, d_out, padding=False):
        super(UNet, self).__init__()

        # Input layer
        self.inp = DoubleConv(d_in, 64)

        # Layers of the encoder path consisiting four Down modules with
        # max pooling in between each module to downsample the input
        # by a factor of 2 each time.
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Layers of the decoder path consisiting four Up modules with
        # ConvTranspose2d in between each module to upsample the input
        # by a factor of 2 each time.
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Output layer
        self.out = nn.Conv2d(64, d_out, kernel_size=1)

    def forward(self, x):  # sourcery skip: inline-immediately-returned-variable
        # Pass through input layer
        x1 = self.inp(x)

        # Encode (downsample) the input by passing it through the
        # encoder path consisting of four Down modules.
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decode (upsample) the input by passing it through the
        # decoder path consisting of four Up modules.
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Pass through output layer
        y = self.out(x)

        return y
