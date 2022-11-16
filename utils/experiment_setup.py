import os, sys
if os.path.dirname(os.path.dirname(os.path.realpath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from data.cifar import CIFARDataModule
from data.svhn import  SVHNDataModule
from data.ood_datamodule import OODDataModule
from models.backbones.resnet import ResNet
from models.backbones.wideresnet import WideResNet
from models.vanilla import Vanilla
from models.regmixup import RegMixup
from argparse import ArgumentParser
from models.ood_evaluable_module import OODEvaluable


def prepare_args():
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument('--model_type', type=str, default='resnet50_vanilla')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--resume_training', action="store_true")
    parser.add_argument('--cross_valid', action="store_true")
    parser.add_argument("--network_category", type=str, choices = ["cnn"]) #default="wideresnet_vanilla", help="Backbone to be used")
    parser.add_argument("--backbone", type=str, default="wideresnet_vanilla", help="Backbone to be used")
    parser.add_argument("--valid_size", type=float, default=None, help="If None, sets valid size 10%, can specify any amount, if 0%  uses test set")
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128), this is the total batch size of all GPUs on the current node'
                             ' when using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--nesterov', action="store_false", help='nesterov momentum')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--sn', default=None, type=float, help='spectral norm factor')
    parser.add_argument('--sr', default=None, type=float, help='stable rank factor')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='whether to use standard augmentation (default: True)')
    parser.add_argument('--name', default='results', type=str,
                        help='name of experiment')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_bins', default=15, type=int,
                        help='Number of bins for reliability plots')
    parser.add_argument("--dataset_directory", type=str, default="../data/", help="Base path for datasets")
    parser.add_argument("--postfix", type=str, default="", help="Postfix")
    parser.add_argument("--lr-scheduler-gamma", default=1e-1, type=float, metavar="LRG",
                        help="Factor by which the learning rate is reduced at each milestone. Default 0.1 for resnet50, set 0.2 for WideResnet",
                        dest="lr_scheduler_gamma")
    parser.add_argument("--num-workers", default=8, type=int, metavar="W", help="number of CPU workers",
                        dest="num_workers")
    parser.add_argument("--milestones", default=[150, 250], type=int, metavar="M", nargs='+',
                        help="Milestones for the scheduler. Default [150,250] for resnet50, set [60,120,160) for WideResNet")

    parser.add_argument('--without_bias', action='store_true')
    parser.add_argument('--eps', default=1e-12, type=float, help='eps for numerical stability')
    parser.add_argument("--class_head_name", default="fc", type=str,
                        help="Name of the layer performing the embeddings to logit transformation, used for checkpoint loading")


    temp_args, _ = parser.parse_known_args()

    #add trainer specific args
    parser = Trainer.add_argparse_args(parser)

    # add model specific args
    if temp_args.backbone.find("wideresnet") >= 0:
        parser = WideResNet.add_model_specific_args(parser)
    elif temp_args.backbone.find("resnet") >= 0:
        parser = ResNet.add_model_specific_args(parser)
    else:
        print("Choose one of the available backbones: wideresnet or resnet")
        sys.exit()

    base_class = OODEvaluable
    network_wrapper = Vanilla

    if temp_args.model_type.find("vanilla") >= 0:
        selected_class = network_wrapper(base_class)
    elif temp_args.model_type.find("regmixup") >= 0:
        selected_class = RegMixup(network_wrapper(base_class))

    parser = selected_class.add_model_specific_args(parser)

    args = parser.parse_args()
    if args.debug:
        from utils import debug
    return selected_class, args


import pytorch_lightning as pl
import torch.nn as nn
import torch

def prepare_trainer(model_class, args):
    dict_args = vars(args)
    pl.utilities.seed.seed_everything(dict_args["seed"])


    if dict_args["cross_valid"] and dict_args['valid_size'] is None: #if crossvalidate but didn't specify a size, use default
        valid_size = 0.1
    elif dict_args["cross_valid"] and dict_args['valid_size'] is not None: #if crossvalidate and specified size, use it; if valid size = 0 and cross_valid is active then cross_validates on the test set
        valid_size = dict_args['valid_size']
    else:  #use test set
        valid_size = 0.

    dm = CIFARDataModule(dict_args["dataset_directory"],
                         dict_args["dataset"],
                         dict_args["augment"],
                         dict_args["batch_size"],
                         dict_args["num_workers"],
                         valid_size = valid_size)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        dirpath=dict_args["default_root_dir"],
        filename="complete_debug_vanilla",
        save_top_k=3,
        # save_last=True,
        mode='max',
    )
    if dict_args["resume_training"]:
        print('Resume training')
        try:
            model = model_class.load_from_checkpoint(checkpoint_path=dict_args["default_root_dir"] + "/last_full_epoch.ckpt",**dict_args, dm=dm)
            trainer = Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], resume_from_checkpoint = dict_args["default_root_dir"] + "/last_full_epoch.ckpt")
        except Exception as e:
            print("Checkpoint not found, reinitializing the network")
            sys.exit()
    else:
        model = model_class(**dict_args, dm=dm)
        trainer = Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
                                             #resume_from_checkpoint=checkpoint_callback.last_model_path)

    return model, trainer, dm
