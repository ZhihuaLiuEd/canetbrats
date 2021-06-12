from __future__ import division
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import BatchNorm3d
from collections import OrderedDict

from .backbone import Backbone
from .featureinteractiongraph import FeatureInteractionGraph
from .convcontextbranch import ConvContextBranch, normal_conv_blocks
from .cgacrf import CGACRF

import bratsUtils

__all__ = ['CANetOutput', 'CANet']

id = random.getrandbits(64)

#restore experiment
#VALIDATE_ALL = False
#PREDICT = True
#RESTORE_ID = 10452690098638733452
#RESTORE_EPOCH = 199

#VISUALIZE_PROB_MAP = True

#general settings
SAVE_CHECKPOINTS = True #set to true to create a checkpoint at every epoch
EXPERIMENT_TAGS = ["bugfreeFinalDrop"]
EXPERIMENT_NAME = "CANet on BraTS19"
EPOCHS = 200
BATCH_SIZE = 1
VIRTUAL_BATCHSIZE = 1
VALIDATE_EVERY_K_EPOCHS = 1
INPLACE = True

#hyperparameters
CHANNELS = 32
INITIAL_LR = 1e-4
L2_REGULARIZER = 1e-5

#logging settings
LOG_EVERY_K_ITERATIONS = 50 #0 to disable logging
LOG_MEMORY_EVERY_K_ITERATIONS = False
LOG_MEMORY_EVERY_EPOCH = True
LOG_EPOCH_TIME = True
LOG_VALIDATION_TIME = True
LOG_HAUSDORFF_EVERY_K_EPOCHS = 0 #must be a multiple of VALIDATE_EVERY_K_EPOCHS
LOG_COMETML = False
LOG_PARAMCOUNT = True
LOG_LR_EVERY_EPOCH = True

#data and augmentation
TRAIN_ORIGINAL_CLASSES = False #train on original 5 classes
DATASET_WORKERS = 1
SOFT_AUGMENTATION = False #Soft augmetation directly works on the 3 classes. Hard augmentation augments on the 5 orignal labels, then takes the argmax
NN_AUGMENTATION = True #Has priority over soft/hard augmentation. Uses nearest-neighbor interpolation
DO_ROTATE = True
DO_SCALE = True
DO_FLIP = True
DO_ELASTIC_AUG = True
DO_INTENSITY_SHIFT = True
RANDOM_CROP = [128, 128, 128]
ROT_DEGREES = 20
SCALE_FACTOR = 1.1
SIGMA = 10
MAX_INTENSITY_SHIFT = 0.1

if not LOG_COMETML:
    experiment = None

if TRAIN_ORIGINAL_CLASSES:
    loss = bratsUtils.bratsDiceLossOriginal5
else:
    def loss(outputs, labels):
        return bratsUtils.bratsDiceLoss(outputs, labels, nonSquared=True)

class CANetOutput(Backbone):

    def __init__(self, backbone):
        super(CANetOutput, self).__init__(backbone)
        self.seg_prob = CANet(240)

    def forward(self, x):

        x1, x2, x3, x4 = self.backbone_forward(x)

        x = self.seg_prob(x1, x2, x3, x4)

        return x

class CANet(nn.Module):
    def __init__(self, in_channels):
        super(CANet, self).__init__()
        inter_channels = in_channels // 2
        self.conv5a = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    BatchNorm3d(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    BatchNorm3d(inter_channels),
                                    nn.ReLU())

        self.gcn = nn.Sequential(OrderedDict([("FeatureInteractionGraph%02d" % i,
                                               FeatureInteractionGraph(inter_channels, 30, kernel=1)
                                               ) for i in range(1)]))

        self.dcn = nn.Sequential(OrderedDict([("ConvContextBranch%02d" % i, ConvContextBranch()) for i in range(1)]))

        self.crffusion_1 = CGACRF(inter_channels, inter_channels, inter_channels)
        self.crffusion_2 = CGACRF(inter_channels, inter_channels, inter_channels)
        self.crffusion_3 = CGACRF(inter_channels, inter_channels, inter_channels)
        self.crffusion_4 = CGACRF(inter_channels, inter_channels, inter_channels)
        self.crffusion_5 = CGACRF(inter_channels, inter_channels, inter_channels)

        self.conv51 = normal_conv_blocks(inter_channels, inter_channels)
        self.conv52 = normal_conv_blocks(inter_channels, inter_channels)

        self.upconv1 = normal_conv_blocks(240, 120)
        self.upconv2 = normal_conv_blocks(120, 60)
        self.upconv3 = normal_conv_blocks(120, 60)
        self.upconv4 = normal_conv_blocks(60, 30)
        self.upconv5 = normal_conv_blocks(60, 30)
        self.upconv6 = normal_conv_blocks(30, 30)
        self.final_conv = nn.Conv3d(30, 3, kernel_size=1, bias=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x1, x2, x3, x4):

        feat_gcn = self.gcn(x3)
        gcn_conv = self.conv51(feat_gcn)

        feat_dcn = self.dcn(x4)
        fcn_conv = self.conv52(feat_dcn)

        conv_hidden = self.crffusion_1(gcn_conv, fcn_conv)
        conv_hidden = self.crffusion_2(gcn_conv, conv_hidden)
        conv_hidden = self.crffusion_3(gcn_conv, conv_hidden)
        conv_hidden = self.crffusion_4(gcn_conv, conv_hidden)
        conv_hidden = self.crffusion_5(gcn_conv, conv_hidden)

        x = torch.cat([x3, conv_hidden], dim=1)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upsample(x)

        x = torch.cat([x2, x], dim=1)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upsample(x)

        x = torch.cat([x1, x], dim=1)
        x = self.upconv5(x)
        x = self.upconv6(x)

        final_conv_output = self.final_conv(x)
        out = torch.sigmoid(final_conv_output)

        return out

net = CANetOutput(backbone='unet_encoder')
optimizer = optim.Adam(net.parameters(), lr=INITIAL_LR, weight_decay=L2_REGULARIZER)
lr_sheudler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 130, 160], 0.2)
