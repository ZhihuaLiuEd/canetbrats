import torch
import torch.nn as nn
from .group_norm import GroupNorm3d as groupnorm

__all__ = ['double_conv', 'single_conv', 'UNetEncoder', 'unet_encoder']

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        groupnorm(out_channels, 1),
        nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        groupnorm(out_channels, 1),
        nn.LeakyReLU(negative_slope=1e-2, inplace=True),
    )

def single_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        groupnorm(out_channels, 1),
        nn.LeakyReLU(negative_slope=1e-2, inplace=True),
    )

class UNetEncoder(nn.Module):
    def __init__(self, n_class=3):
        super(UNetEncoder, self).__init__()
        self.dconv_down1 = double_conv(4, 30)
        self.dconv_down2 = double_conv(30, 60)
        self.dconv_down3 = double_conv(60, 120)
        self.dconv_down4 = double_conv(120, 240)
        self.dconv_down5 = single_conv(240, 480)
        self.dconv_down6 = single_conv(480, 240)

        self.maxpool = nn.MaxPool3d(2)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.dconv_up4_1 = single_conv(240 + 240, 240)
        self.dconv_up4_2 = single_conv(240, 120)

        self.dconv_up3_1 = single_conv(120 + 120, 120)
        self.dconv_up3_2 = single_conv(120, 60)

        self.dconv_up2_1 = single_conv(60 + 60, 60)
        self.dconv_up2_2 = single_conv(60, 30)

        self.dconv_up1_1 = single_conv(30 + 30, 30)
        self.dconv_up1_2 = single_conv(30, 30)

        self.conv_last = nn.Conv3d(30, n_class, 1)


    def forward(self, x):

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)
        conv6 = self.dconv_down6(conv5)

        x = self.upsample(conv6)
        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up4_1(x)
        x = self.dconv_up4_2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3_1(x)
        x = self.dconv_up3_2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2_1(x)
        x = self.dconv_up2_2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1_1(x)
        x = self.dconv_up1_2(x)

        out = self.conv_last(x)

        return out

def unet_encoder():
    model = UNetEncoder()
    return model

