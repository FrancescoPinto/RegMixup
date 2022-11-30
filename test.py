import os, sys
if os.path.dirname(os.path.dirname(os.path.realpath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.experiment_setup import prepare_args, prepare_trainer
from data.cifar_c_datamodule import CIFAR_C_DataModule
from data.ood_datamodule import OODDataModule
from data.cifar import CIFARDataModule, Cifar10v6TestDataset, Cifar10_2TestDataset
from data.svhn import SVHNDataModule
import torchvision

def simple_test(trainer, model, dm):
    trainer.test(model, dm.test_dataloader())

def test_ood(trainer, model):
    ind_dm = CIFARDataModule(model.hparams.dataset_directory,
                               model.hparams.dataset,
                               model.hparams.augment,
                               model.hparams.batch_size,
                               model.hparams.num_workers,
                             )

    ood_dms = {
        "cifar": CIFARDataModule(model.hparams.dataset_directory,
                                 "cifar10" if model.hparams.dataset == "cifar100" else "cifar100",
                                 model.hparams.augment,
                                 model.hparams.batch_size,
                                 model.hparams.num_workers,
                                 test_transforms=ind_dm.test_transforms),  # normalized wrt IND data
        "svhn": SVHNDataModule(model.hparams.dataset_directory, model.hparams.batch_size, model.hparams.num_workers,
                               test_transforms=ind_dm.test_transforms),  # normalized wrt IND data)

    }
    ood_dm = OODDataModule(ind_datamodule = ind_dm, ind_datamodule_name = model.hparams.dataset, ood_datamodules_dict= ood_dms)
    ood_dm.prepare_data()
    ood_dm.setup("test")


    model.evaluate_ood_mode = True
    model.ood_dm = ood_dm
    trainer.test(model, ood_dm.test_dataloader())
    model.evaluate_ood_mode = False


def test_data_shift(trainer, model, dm, args):
    # problema, credo il filtraggio delle label nell'output' vada fatto dinamicamente (come fa hendryicks nei suoi repository, quello spieg l)
    datashift_datasets = {
        "cifar10-v6": Cifar10v6TestDataset(args.dataset_directory,
                                           test_transforms=dm.test_transforms, batch_size=args.batch_size,
                                           num_workers=args.num_workers ),
        "cifar10.2": Cifar10_2TestDataset(args.dataset_directory,
                                           test_transforms=dm.test_transforms, batch_size=args.batch_size,
                                           num_workers=args.num_workers),
        "cifar-c": CIFAR_C_DataModule.get_composite_data_shift_module(args.dataset_directory, args.dataset,
                                                      dm.test_transforms, args.batch_size,
                                                      args.num_workers),

    }

    for name, d in datashift_datasets.items():
        model.evaluate_datashift_mode = True
        model.datashift_name = name
        d.setup("test")
        model.datashift_dm = d
        trainer.test(model, d.test_dataloader())
        model.evaluate_corruptions_mode = False
        #inside decides whether to use composite or not



def test(model_class, args):
    model, trainer, dm = prepare_trainer(model_class, args)
    dm.prepare_data()
    dm.setup("test")
    simple_test(trainer, model, dm)
    test_ood(trainer,model)
    test_data_shift(trainer,model,dm,args)




if __name__ == '__main__':
    model_class, args = prepare_args()
    test(model_class, args)
