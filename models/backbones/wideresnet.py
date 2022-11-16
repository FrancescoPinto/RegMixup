import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
if os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from models.regularizers.spectral_normalization import spectral_norm
from models.regularizers.stable_rank import stable_rank
import numpy as np
import tqdm
from argparse import ArgumentParser


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, sn_factor = None, sr_factor = None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        assert(sn_factor is None or sr_factor is None)
        if sn_factor is not None:
            self.conv1 = spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False), sn_factor = sn_factor)
        elif sr_factor is not None:
            self.conv1 = stable_rank(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False), rank = sr_factor)
        else:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)


        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)

        if sn_factor is not None:
            self.conv2 = spectral_norm(nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                                                 padding=1, bias=False), sn_factor=sn_factor)
        elif sr_factor is not None:
            self.conv2 = stable_rank(nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                                                 padding=1, bias=False), rank = sr_factor)
        else:
            self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                                                 padding=1, bias=False)

        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        if not self.equalInOut:
            if sn_factor is not None:
                self.convShortcut = spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                            padding=0, bias=False), sn_factor=sn_factor)
            elif sr_factor is not None:
                self.convShortcut = stable_rank(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                            padding=0, bias=False), rank=sr_factor)
            else:
                self.convShortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                            padding=0, bias=False)


    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, sn_factor = None, sr_factor = None):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, sn_factor, sr_factor)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, sn_factor, sr_factor):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, sn_factor, sr_factor))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, sn_factor = None, sr_factor = None, gp_hyperparams=None, model_type = "wideresnet_vanilla", tsne_embedding_dim=2, without_bias=False, std_init_layers=1.,
                 densqrt=False):
        super(WideResNet, self).__init__()

        self.model_type = model_type
        self.without_bias = without_bias

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        if sn_factor is not None:
            self.conv1 = spectral_norm(nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                                                 padding=1, bias=False), sn_factor=sn_factor)
        elif sr_factor is not None:
            self.conv1 = stable_rank(nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                                                 padding=1, bias=False), rank=sr_factor)
        else:
            self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                                                 padding=1, bias=False)

        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, sn_factor=sn_factor, sr_factor=sr_factor)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, sn_factor=sn_factor, sr_factor=sr_factor)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, sn_factor=sn_factor, sr_factor=sr_factor)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and not without_bias:
                m.bias.data.zero_()

        self.nChannels = nChannels[3]
        self.embedding_size = self.nChannels #embedding size


    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        embedding = out.view(-1, self.nChannels)
        return embedding




    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser],add_help=False)
        parser.add_argument('--layers', default=28, type=int,
                            help='total number of layers (default: 28)')
        parser.add_argument('--widen-factor', default=10, type=int,
                            help='widen factor (default: 10)')
        parser.add_argument('--droprate', default=0.1, type=float,
                            help='dropout probability (default: 0.0)')

        return parser




