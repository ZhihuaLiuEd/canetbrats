import torch
import torch.nn as nn
from .group_norm import GroupNorm3d as groupnorm

__all__ = ['double_conv', 'single_conv', 'UNetEncoder', 'unet_encoder']

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        groupnorm(in_channels, 8),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        groupnorm(out_channels, 8),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
    )

def single_conv(in_channels, out_channels):
    return nn.Sequential(
        groupnorm(in_channels, 8),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
    )

class UNetEncoder(nn.Module):
    def __init__(self, n_class=3):
        super(UNetEncoder, self).__init__()
        self.init_conv = nn.Conv3d(4, 32, kernel_size=3, stride=1, padding=1)
        self.dropout3d = nn.Dropout3d(p=0.2)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)
        self.dconv_down4 = double_conv(128, 256)
        self.dconv_down5 = single_conv(256, 512)
        self.dconv_down6 = single_conv(512, 256)

        self.maxpool = nn.MaxPool3d(2)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.dconv_up4_1 = single_conv(256 + 256, 256)
        self.dconv_up4_2 = single_conv(256, 128)

        self.dconv_up3_1 = single_conv(128 + 128, 128)
        self.dconv_up3_2 = single_conv(128, 64)

        self.dconv_up2_1 = single_conv(64 + 64, 64)
        self.dconv_up2_2 = single_conv(64, 32)

        self.dconv_up1_1 = single_conv(32 + 32, 32)
        self.dconv_up1_2 = single_conv(32, 32)

        self.conv_last = nn.Conv3d(32, n_class, 1)

    def forward(self, x):

        conv1 = self.init_conv(x)
        x = self.dropout3d(x)

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

