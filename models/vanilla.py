import os, sys
if os.path.dirname(os.path.dirname(os.path.realpath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from pathlib import Path
from typing import Union

import torch
from torch import nn, optim
import torch.nn.functional as F
from utils.backbones_factory import get_model
from metrics.ECE import ECE
#  --- Pytorch-lightning module ---
import pytorch_lightning as pl
from argparse import ArgumentParser

def Vanilla(class_to_extend):
    class VanillaModel(class_to_extend):

        @classmethod
        def add_model_specific_args(cls, parent_parser):  # pragma: no-cover
            parent_parser = super().add_model_specific_args(parent_parser)
            parser = ArgumentParser(parents=[parent_parser],add_help=False)
            return parser

        def __init__(
            self,
            *args,
            **kwargs,
        ) -> None:
            """
            Args:
                dl_path: Path where the data will be downloaded
            """
            super().__init__(*args,**kwargs)
            self.build_model()
            self.build_loss()
            self.build_nll()

            self.output_is_logits = True

        def build_loss(self):
            self.loss = F.cross_entropy

        def build_nll(self):
            self.nll = F.cross_entropy

        def build_model(self):
            # 1. Prepare backbone:
            self.feature_extractor = get_model(self.hparams)
            # 2. Classifier:
            self.fc = nn.Linear(self.feature_extractor.embedding_size, self.dm.num_classes, bias=not self.hparams.without_bias)

            if self.hparams.resume_training:
                if os.path.exists(self.hparams.default_root_dir + "/last_full_epoch.ckpt"):
                    cpt = torch.load(self.hparams.default_root_dir + "/last_full_epoch.ckpt")
                    fc_dict = {'weight': cpt['state_dict'].pop(f'{self.hparams.class_head_name}.weight'), "bias":cpt['state_dict'].pop(f'{self.hparams.class_head_name}.bias')}
                    new_feature_dict = {}
                    for k,v in cpt["state_dict"].items():
                        new_feature_dict[k.replace("feature_extractor.",'')] = v

                    self.feature_extractor.load_state_dict(new_feature_dict)
                    self.fc.load_state_dict(fc_dict)



        def forward(self, x):
            """Forward pass. Returns logits."""
            # 1. Feature extraction:
            embeddings = self.feature_extractor(x)

            # 2. Classifier (returns logits):
            x = self.fc(embeddings)

            return x, embeddings

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), self.hparams.lr,
                                        momentum=self.hparams.momentum, nesterov=self.hparams.nesterov,
                                        weight_decay=self.hparams.weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.milestones,
                                                             gamma=self.hparams.lr_scheduler_gamma)

            return [optimizer], [scheduler]


        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat, _ = self(x)
            loss = self.loss(y_hat, y)

            if self.hparams.accelerator == "dp": #dp has bug, if you don't return only loss it gets loss = nan
                return loss
            else:
                return {"loss":loss, "y_logits":y_hat, "y_true":y}

        def training_epoch_end(self, outputs):
            print("Epoch, saving because epoch ended: ", str(self.trainer.current_epoch))
            print("SAVING CHECKPOINT HERE: ", self.hparams["default_root_dir"] + "/last_full_epoch.ckpt")
            self.hparams.dm = None
            self.trainer.save_checkpoint(self.hparams["default_root_dir"] + "/last_full_epoch.ckpt")


    return VanillaModel
