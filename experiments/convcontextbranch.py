import torch
from .group_norm import GroupNorm3d as groupnorm

torch_ver = torch.__version__[:3]

__all__ = ['ConvContextBranch']

import torch
import torch.nn as nn

def normal_conv_blocks(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        groupnorm(out_channels, 1),
        nn.LeakyReLU(negative_slope=1e-2, inplace=True),
    )

class ConvContextBranch(nn.Module):
    def __init__(self):
        super(ConvContextBranch, self).__init__()
        self.dconv_down1 = normal_conv_blocks(in_channels=240, out_channels=480)
        self.dconv_down2 = normal_conv_blocks(in_channels=480, out_channels=240)
        self.dconv_up1 = normal_conv_blocks(in_channels=480, out_channels=240)
        self.dconv_up2 = normal_conv_blocks(in_channels=240, out_channels=120)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x):
        x_mp = self.maxpool(x)
        conv1 = self.dconv_down1(x_mp)
        conv2 = self.dconv_down2(conv1)
        conv2_up = self.upsample(conv2)
        conv2_up_concat = torch.cat([conv2_up, x], dim=1)
        conv3 = self.dconv_up1(conv2_up_concat)
        conv4 = self.dconv_up2(conv3)
        out = self.upsample(conv4)

        return out


