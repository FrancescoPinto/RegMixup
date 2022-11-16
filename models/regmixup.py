import os, sys
if os.path.dirname(os.path.dirname(os.path.realpath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from argparse import ArgumentParser
import torch
from models.regularizers.mixup import mixup_data
import torch.nn.functional as F


def regmixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def RegMixup(model_to_extend):
    class RegMixupModel(model_to_extend):
        @classmethod
        def add_model_specific_args(cls,parent_parser):  # pragma: no-cover
            parent_parser = super().add_model_specific_args(parent_parser)
            parser = ArgumentParser(parents=[parent_parser],add_help=False)
            parser.add_argument('--mixup_alpha', default=20., type=float)
            parser.add_argument('--mixup_beta', default=20., type=float)
            parser.add_argument('--loss', default="", type=str)

            return parser


        def build_loss(self):
            self.loss = regmixup_criterion

        def training_step(self, batch, batch_idx):
            x, y = batch
            mixup_x, part_y_a, part_y_b, lam = mixup_data(x, y, self.hparams.mixup_alpha, self.hparams.mixup_beta)

            targets_a = torch.cat([y, part_y_a])
            targets_b = torch.cat([y,part_y_b])
            x = torch.cat([x, mixup_x], dim=0)

            logits, embeddings = self(x)
            loss = self.loss(F.cross_entropy, logits, targets_a,targets_b, lam)

            return loss
    return RegMixupModel

