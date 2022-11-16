import os, sys
if os.path.dirname(os.path.dirname(os.path.realpath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.experiment_setup import prepare_args, prepare_trainer

import sys
# from data.real.cifar_c_datamodule import get_corruptions_datamodule



def train(result):
    model_class, args = result
    model, trainer, dm = prepare_trainer(model_class, args)
    trainer.fit(model, dm)

if __name__ == '__main__':
    train(prepare_args())
