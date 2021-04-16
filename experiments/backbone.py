import torch.nn as nn
from .backbone_resnet import *
from .backbone_unet_encoder import unet_encoder

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

__all__ = ['Backbone']

class Backbone(nn.Module):
    def __init__(self, nclass, backbone):
        super(Backbone, self).__init__()
        self.nclass = nclass

        if backbone == 'resnet3d18':
            self.pretrained = resnet3d18()
        elif backbone == 'resnet3d34':
            self.pretrained = resnet3d34()
        elif backbone == 'resnet3d50':
            self.pretrained = resnet3d50()
        elif backbone == 'unet_encoder':
            self.pretrained = unet_encoder()
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        #upsample options
        self._up_kwargs = up_kwargs

    def backbone_forward(self, x):
        conv1 = self.pretrained.dconv_down1(x)
        x1 = self.pretrained.maxpool(conv1)

        conv2 = self.pretrained.dconv_down2(x1)
        x2 = self.pretrained.maxpool(conv2)

        conv3 = self.pretrained.dconv_down3(x2)
        x3 = self.pretrained.maxpool(conv3)

        conv4 = self.pretrained.dconv_down4(x3)

        return conv1, conv2, conv3, conv4
