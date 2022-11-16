import os, sys
if os.path.dirname(os.path.dirname(os.path.realpath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


from models.backbones.resnet import resnet50
from models.backbones.wideresnet import WideResNet
import torchvision
import timm


def get_model(hp):
    if hp.dataset == "cifar10":
        num_classes = 10
    elif hp.dataset == "cifar100":
        num_classes = 100

    if hp.backbone == "wideresnet_vanilla":
        return WideResNet(hp.layers, num_classes, hp.widen_factor, dropRate=hp.droprate, model_type=hp.model_type, without_bias=hp.without_bias)
    elif hp.backbone == "wideresnet_vanilla_sn":
        return WideResNet(hp.layers, num_classes, hp.widen_factor, dropRate=hp.droprate, sn_factor = hp.sn, model_type=hp.model_type, without_bias=hp.without_bias)
    elif hp.backbone == "wideresnet_vanilla_sr":
        return WideResNet(hp.layers, num_classes, hp.widen_factor, dropRate=hp.droprate, sr_factor = hp.sr, model_type=hp.model_type, without_bias=hp.without_bias)
    elif hp.backbone == "resnet50_vanilla":
        return resnet50(num_classes = num_classes, model_type=hp.model_type, without_bias=hp.without_bias)
    elif hp.backbone == "resnet50_vanilla_sn":
        return resnet50(num_classes = num_classes, sn_factor = hp.sn, model_type=hp.model_type, without_bias=hp.without_bias)
    elif hp.backbone == "resnet50_vanilla_sr":
         return resnet50(num_classes=num_classes, sr_factor = hp.sr, model_type=hp.model_type, without_bias=hp.without_bias)

