import math
import torch
from torch.nn import Conv3d, Module, Sigmoid
from torch.nn import BatchNorm3d

torch_ver = torch.__version__[:3]

__all__ = ['CGACRF']

def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, Conv3d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class CGACRF(Module):
    """
    Meanfield updating for the features and the attention for one pair of features.
    bottom_list is a list of observation features derived from the backbone CNN.
    update attention map
    a_s <-- y_s * (K_s conv y_S)
    a_s = b_s conv a_s
    a_s <-- Sigmoid(-(a_s + a_s))
    update the last scale feature map y_S
    y_s <-- K conv y_s
    y_S <-- x_S + (a_s * y_s)
    """

    def __init__(self, bottom_send, bottom_receive, feat_num):
        super(CGACRF, self).__init__()

        self.atten = Conv3d(in_channels=bottom_send + bottom_receive, out_channels=feat_num,
                              kernel_size=3, stride=1, padding=1)
        self.norm_atten = Sigmoid()
        self.message = Conv3d(in_channels=bottom_send, out_channels=feat_num, kernel_size=3,
                                stride=1, padding=1)
        self.scale = Conv3d(in_channels=feat_num, out_channels=bottom_receive, kernel_size=1, bias=True)

    def forward(self, g_s, c_s):
        # x_s -> g_s
        # x_S -> c_s
        # y_s -> g_h
        # y_S -> c_h

        # update attention map
        a_s = torch.cat((g_s, c_s), dim=1)
        a_s = self.atten(a_s)
        a_s = self.norm_atten(a_s)

        # update the last scale feature map y_S
        g_h = self.message(g_s)
        c_h = g_h.mul(a_s)  # production
        c_h = self.scale(c_h) # scale
        c_h = c_s + c_h  # eltwise sum
        return c_h