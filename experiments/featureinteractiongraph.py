# We sincerely thank Dr.Yunpeng Chen for open-sourcing the GloRe:
# Adopted from: https://github.com/cypw/GloRe

import torch
from torch.nn import Conv1d, Conv3d, Module, BatchNorm3d, ReLU
torch_ver = torch.__version__[:3]

__all__ = ['FeatureInteractionGraph']

class GCN(Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.relu(h)
        h = self.conv2(h)
        return h

class FeatureInteractionGraph(Module):
    def __init__(self, num_in, num_mid, stride=(1,1), kernel=1):
        super(FeatureInteractionGraph, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        kernel_size = (kernel, kernel, kernel)
        padding = 1 if kernel == 3 else 0

        # reduce dimension
        self.conv_state = Conv3d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate graph transformation function
        self.conv_proj = Conv3d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)
        # ----------
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # tail: extend dimension
        self.fc_2 = Conv3d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=1,
                              groups=1, bias=False)

        self.blocker = BatchNorm3d(num_in)

    def forward(self, x):
        '''
        :param x: (n, c, h, w)
        '''
        batch_size = x.size(0)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(batch_size, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: pixel space -> instance space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: instance space -> pixel space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(batch_size, self.num_s, *x.size()[2:])

        # -----------------
        # final
        out = x + self.blocker(self.fc_2(x_state))

        return out