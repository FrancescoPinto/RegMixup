'''
Pytorch implementation of ResNet models.

Reference:
[1] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR, 2016.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
if os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from models.regularizers.spectral_normalization import spectral_norm
from models.regularizers.stable_rank import stable_rank

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, sn_factor=None, sr_factor=None):
        super(BasicBlock, self).__init__()
        assert (sn_factor is None or sr_factor is None)
        if sn_factor is not None:
            self.conv1 = spectral_norm(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False), sn_factor=sn_factor)
        elif sr_factor is not None:
            self.conv1 = stable_rank(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False), rank=sr_factor)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        if sn_factor is not None:
            self.conv2 = spectral_norm(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False), sn_factor=sn_factor)
        elif sr_factor is not None:
            self.conv2 = stable_rank(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False), rank=sr_factor)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            if sn_factor is not None:
                temp_conv = spectral_norm(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    sn_factor=sn_factor)
            elif sr_factor is not None:
                temp_conv = stable_rank(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    rank=sr_factor)
            else:
                temp_conv = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

            self.shortcut = nn.Sequential(
                temp_conv,
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, sn_factor=None, sr_factor=None):
        super(Bottleneck, self).__init__()
        assert (sn_factor is None or sr_factor is None)
        if sn_factor is not None:
            self.conv1 = spectral_norm(
                nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
                sn_factor=sn_factor)
        elif sr_factor is not None:
            self.conv1 = stable_rank(
                nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
                rank=sr_factor)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        if sn_factor is not None:
            self.conv2 = spectral_norm(
                nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                sn_factor=sn_factor)
        elif sr_factor is not None:
            self.conv2 = stable_rank(
                nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                rank=sr_factor)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        if sn_factor is not None:
            self.conv3 = spectral_norm(
                nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False),
                sn_factor=sn_factor)
        elif sr_factor is not None:
            self.conv3 = stable_rank(
                nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False),
                rank=sr_factor)
        else:
            self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:

            if sn_factor is not None:
                temp_conv = spectral_norm(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    sn_factor=sn_factor)
            elif sr_factor is not None:
                temp_conv = stable_rank(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    rank=sr_factor)
            else:
                temp_conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)

            self.shortcut = nn.Sequential(
                temp_conv,
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, temp=1.0, sn_factor = None, sr_factor = None, model_type = None,
                 gp_hyperparams = None, without_bias = False, densqrt=False):
        super(ResNet, self).__init__()

        self.model_type = model_type
        self.without_bias = without_bias

        self.in_planes = 64

        if sn_factor is not None:
            self.conv1 = spectral_norm(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                sn_factor=sn_factor)
        elif sr_factor is not None:
            self.conv1 = stable_rank(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                rank=sr_factor)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, sn_factor=sn_factor, sr_factor=sr_factor)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, sn_factor=sn_factor, sr_factor=sr_factor)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, sn_factor=sn_factor, sr_factor=sr_factor)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, sn_factor=sn_factor, sr_factor=sr_factor)
        self.embedding_size = 512 * block.expansion #embedding size


    def _make_layer(self, block, planes, num_blocks, stride, sn_factor, sr_factor):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, sn_factor=sn_factor, sr_factor=sr_factor))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        embeddings = out.view(out.size(0), -1)
        return embeddings
    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        print(parent_parser)
        return parent_parser


def resnet18(temp=1.0,  sn_factor =None, sr_factor=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], temp=temp,sn_factor=sn_factor, sr_factor=sr_factor, **kwargs)
    return model


def resnet34(temp=1.0, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], temp=temp, **kwargs)
    return model


def resnet50(temp=1.0, sn_factor =None, sr_factor=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], temp=temp, sn_factor=sn_factor, sr_factor=sr_factor, **kwargs)
    return model


def resnet101(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], temp=temp, **kwargs)
    return model


def resnet110(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 26, 3], temp=temp, **kwargs)
    return model


def resnet152(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], temp=temp, **kwargs)
    return model