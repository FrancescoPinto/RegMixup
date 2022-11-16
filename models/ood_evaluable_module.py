import os, sys
if os.path.dirname(os.path.dirname(os.path.realpath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from metrics.ECE import ECE
from metrics.AdaECE import AdaECE



import pytorch_lightning as pl
from utils.ood_eval_utils import default_reduce_outputs, entropy, confidence, dempster_shafer_metric
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import sklearn.metrics as sk
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.reliability_plots import reliability_plot
from data.datashift_datamodule import CompositeDataShiftDataModule
# \\
#
# DO delaunay on the means (10^8 should be affordable, or further increase the pca threshold)
from torchmetrics import Accuracy
class OODEvaluable(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        return parent_parser

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            dl_path: Path where the data will be downloaded
        """
        super().__init__()
        self.save_hyperparameters()

        Path(self.hparams.default_root_dir).mkdir(parents=True, exist_ok=True)
        Path(self.hparams.default_root_dir + "/"+self.hparams.name).mkdir(parents=True, exist_ok=True)

        with open(self.hparams.default_root_dir + '/train.sh', 'w') as f:
            command = "python train.py "
            command += ' '.join(sys.argv[1:])
            f.write(command)

        self.laplace_postfix=""
        #now everything is accessible from self.hparams
        print(self.hparams)
        self.dm = kwargs["dm"]

        self.valid_metrics = torch.nn.ModuleDict({  #todo, add ood metrics
            'val_accuracy': Accuracy(),
            'val_ece': ECE(process_logits=False),
            'val_adaece': AdaECE(process_logits=False)
        })

        self.test_metrics = torch.nn.ModuleDict({  #todo, add ood metrics
            'test_accuracy': Accuracy(),
            'test_ece': ECE(process_logits=False),
            'test_adaece':AdaECE(process_logits=False)
        })

        self.train_metrics = torch.nn.ModuleDict({
            'train_accuracy': Accuracy(),
            'train_ece': ECE(process_logits=False),
            'train_adaece': AdaECE(process_logits=False)

        })

        self.save_hyperparameters()
        self.output_ood_scores = { #1 indicates IND has high value, OOD low value; 0 indicates IND has low value, OOD high
                              "unc_confidence":(confidence,1),"unc_dempster_shafer": (dempster_shafer_metric, 0), "unc_entropy": (entropy,0)}



        self.evaluate_corruptions_mode = False
        self.evaluate_datashift_mode = False
        self.evaluate_ood_mode = False

        self.val_counter = 0 #if you want to compute validation in val_step_end,
                            #this is needed to consider for the val metrics ONLY the IND dataset
        self.corrupted_dm = None
        self.firstk_projmat = None
        self.train_mean_embeds = None
        self.train_std_embeds = None
        self.train_acc_eval = False



    def validation_step(self, batch, batch_idx):
        x, y = batch
        # import pdb; pdb.set_trace()
        y_hat, embeds = self(x)

        result = {"y_logits":y_hat, "y_true": y , "embeds": embeds}
        result = self.add_uncertainty_scores(y_hat, embeds, result, "val", None)
        return result

    def validation_step_end(self, outputs: dict) -> torch.Tensor:
        return outputs

    def validation_epoch_end(self, outputs):
        aggregated_outputs = self.aggregate_outputs(outputs)
        self.val_counter = 0

        torch.save({"names": [self.dm.dataset_name],
                    # "offsets": self.dm.valid_offsets,
                    "y_logits": aggregated_outputs["y_logits"],
                    "y_true": aggregated_outputs["y_true"],
                    "embeds": aggregated_outputs["embeds"]},
                   f'{self.hparams.default_root_dir}/val_outputs_seed{self.hparams.seed}')

        self.log_metrics(outputs, prefix="val", on_step=False)
        collected_uncertainties = self.collect_uncertainties(aggregated_outputs, prefix="val")


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, embeds = self(x)

        result = {"y_logits":y_hat, "y_true": y, "embeds":embeds}
        result = self.add_uncertainty_scores(y_hat,embeds, result, "test", None)
        return result

    def test_step_end(self, outputs: dict) -> torch.Tensor:
        return outputs
        #return {"y_logits":outputs['y_logits'], "y_true": outputs['y_true'], "nll":outputs["nll"]}




    def test_epoch_end(self, outputs):
        # import pdb; pdb.set_trace()

        self.test_counter = 0

        #aggregate results across multiple batches
        aggregated_outputs = self.aggregate_outputs(outputs)
        #store results (for fast testing, especially ensembles)
        # import pdb; pdb.set_trace()+
        if not hasattr(self.hparams, "current_epoch"):
            if not hasattr(self.trainer, "current_epoch"):
                self.hparams.current_epoch=0
            else:
                self.hparams.current_epoch = self.trainer.current_epoch

        if self.evaluate_datashift_mode:
            print(f"Evaluating test_{self.datashift_name}")
            self.evaluate_datashift(aggregated_outputs, prefix=f"{'train' if self.train_acc_eval else 'test'}_{self.datashift_name}")
            if isinstance(self.datashift_dm, CompositeDataShiftDataModule):
                names = self.datashift_dm.subsets_datasets_names
                offsets = self.datashift_dm.offsets
            else:
                names = self.datashift_name
                offsets = []

            torch.save({"names": names ,
                        "offsets": offsets,
                        "y_logits": aggregated_outputs["y_logits"].cpu(),
                        "y_true": aggregated_outputs["y_true"].cpu()},
                       f'{self.hparams.default_root_dir}/outputs_logits_{"train" if self.train_acc_eval else "test"}_{self.datashift_name}_seed{self.hparams.seed}_{self.laplace_postfix}{self.hparams.current_epoch}')

        elif self.evaluate_ood_mode:
            torch.save({"names": [self.ood_dm.ind_datamodule_name, *self.ood_dm.ood_datamodules_names],
                        "offsets": self.ood_dm.test_offsets,
                        "y_logits": aggregated_outputs["y_logits"].cpu(),
                        "y_true": aggregated_outputs["y_true"].cpu(),
                       "embeds": aggregated_outputs["embeds"].cpu()},
                       f'{self.hparams.default_root_dir}/outputs_ood_{"train" if self.train_acc_eval else "test"}_seed{self.hparams.seed}_{self.hparams.current_epoch}')
            print("SAVING THE RESULTS IN: " + f'{self.hparams.default_root_dir}/outputs_ood_seed{self.hparams.seed}_{self.hparams.current_epoch}')
            collected_uncertainties = self.collect_uncertainties(aggregated_outputs, prefix="test" if not self.train_acc_eval else "train")

            self.test_ood(collected_uncertainties,prefix="test" if not self.train_acc_eval else "train")

        else:
            torch.save({"names": [self.dm.dataset_name],
                        "y_logits": aggregated_outputs["y_logits"].cpu(),
                        "y_true": aggregated_outputs["y_true"].cpu(),
                        "embeds": aggregated_outputs["embeds"].cpu()},
                       f'{self.hparams.default_root_dir}/outputs_{"train" if self.train_acc_eval else "test"}_seed{self.hparams.seed}_{self.hparams.current_epoch}_{self.hparams.dataset}')

            self.log_metrics(outputs, prefix="test" if not self.train_acc_eval else "train", on_step=False)




    def aggregate_outputs(self, outputs):

        aggr = {k: [] for k in outputs[0].keys()}

        for k,v in aggr.items():
            for o in outputs:
                aggr[k].append(o[k])
            aggr[k] = torch.cat(aggr[k], dim=0)
        return aggr

    def evaluate_datashift(self, outputs, prefix):
        result_latex = {}
        acc = Accuracy().cuda()
        if self.hparams.dataset == "cifar10" or self.hparams.dataset == "cifar100" or self.hparams.dataset == "imagenet1k":
            topks = {}
            for top_k in range(2,9):
                topks[top_k]= Accuracy(top_k=top_k).cuda()
        ece = ECE(process_logits=False).cuda()
        adaece = AdaECE(process_logits=False).cuda()

        if isinstance(self.datashift_dm, CompositeDataShiftDataModule):
            offsets = self.datashift_dm.offsets
            accuracies = [acc(self.get_softmax_output(outputs,self.datashift_dm)[0:offsets[0]], outputs['y_true'][0:offsets[0]].int()).unsqueeze(0)]
            if self.hparams.dataset == "cifar10" or self.hparams.dataset == "cifar100" or self.hparams.dataset == "imagenet1k":
                topk_accuracy = {}
                for top_k in range(2,9):
                    topk_accuracy[top_k] = [topks[top_k](self.get_softmax_output(outputs, self.datashift_dm)[0:offsets[0]],
                                         outputs['y_true'][0:offsets[0]].int()).unsqueeze(0).item()]
            eces = [ece(self.get_softmax_output(outputs,self.datashift_dm)[0:offsets[0]], outputs['y_true'][0:offsets[0]].int()).unsqueeze(0)]
            adaeces = [adaece(self.get_softmax_output(outputs,self.datashift_dm)[0:offsets[0]], outputs['y_true'][0:offsets[0]].int()).unsqueeze(0)]
            print(f"{self.datashift_dm.subsets_datasets_names[0]} ",accuracies[-1])
            result_latex[f"{self.datashift_dm.subsets_datasets_names[0]}_{prefix}_accuracy"] = accuracies[-1].item()
            result_latex[f"{self.datashift_dm.subsets_datasets_names[0]}_{prefix}_ece"] = eces[-1].item()
            result_latex[f"{self.datashift_dm.subsets_datasets_names[0]}_{prefix}_adaece"] = adaeces[-1].item()

            for _ in range(len(offsets)-1):
                accuracies.append(acc(self.get_softmax_output(outputs,self.datashift_dm)[offsets[_]:offsets[_+1]], outputs['y_true'][offsets[_]:offsets[_+1]].int()).unsqueeze(0))
                eces.append(ece(self.get_softmax_output(outputs,self.datashift_dm)[offsets[_]:offsets[_+1]], outputs['y_true'][offsets[_]:offsets[_+1]].int()).unsqueeze(0))
                adaeces.append(adaece(self.get_softmax_output(outputs,self.datashift_dm)[offsets[_]:offsets[_+1]], outputs['y_true'][offsets[_]:offsets[_+1]].int()).unsqueeze(0))
                if self.hparams.dataset == "cifar10" or self.hparams.dataset == "cifar100" or self.hparams.dataset == "imagenet1k":
                    for top_k in range(2, 9):
                        topk_accuracy[top_k].append(topks[top_k](self.get_softmax_output(outputs, self.datashift_dm)[offsets[_]:offsets[_+1]],
                                         outputs['y_true'][offsets[_]:offsets[_+1]].int()).unsqueeze(0).item())
                        result_latex[f"{self.datashift_dm.subsets_datasets_names[_+1]}_{prefix}_top{top_k}acc"] = topk_accuracy[top_k][-1].item()

                print(f"{self.datashift_dm.subsets_datasets_names[_+1]} ",accuracies[-1])
                result_latex[f"{self.datashift_dm.subsets_datasets_names[_+1]}_{prefix}_accuracy"] = accuracies[-1].item()
                result_latex[f"{self.datashift_dm.subsets_datasets_names[_+1]}_{prefix}_ece"] = eces[-1].item()
                result_latex[f"{self.datashift_dm.subsets_datasets_names[_+1]}_{prefix}_adaece"] = adaeces[-1].item()


            print(torch.cat(accuracies, dim=0))
            accuracies = torch.cat(accuracies, dim=0).mean()
            eces = torch.cat(eces, dim=0).mean()
            adaeces = torch.cat(adaeces, dim=0).mean()

            if self.hparams.dataset == "cifar10" or self.hparams.dataset == "cifar100" or self.hparams.dataset == "imagenet1k":
                for top_k in range(2, 9):
                    result_latex[f"{prefix}_top{top_k}acc"] =  torch.cat(topk_accuracy[top_k],dim=0).mean()
        else:
            accuracies = acc(self.get_softmax_output(outputs,self.datashift_dm), outputs['y_true'].int()).unsqueeze(0)
            eces = ece(self.get_softmax_output(outputs,self.datashift_dm), outputs['y_true'].int()).unsqueeze(0)
            adaeces = adaece(self.get_softmax_output(outputs,self.datashift_dm), outputs['y_true'].int()).unsqueeze(0)
            if self.hparams.dataset == "cifar10" or self.hparams.dataset == "cifar100" or self.hparams.dataset == "imagenet1k":
                for top_k in range(2, 9):
                    result_latex[f"{prefix}_top{top_k}acc"] = topks[top_k](self.get_softmax_output(outputs,self.datashift_dm), outputs['y_true'].int()).unsqueeze(0).item()


        self.log(f"{prefix}_accuracy", accuracies, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}_ece", eces,prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}_adaece", adaeces,prog_bar=True, on_step=False, on_epoch=True)
        result_latex[f"{prefix}_accuracy"] = accuracies.item()
        result_latex[f"{prefix}_ece"] = eces.item()
        result_latex[f"{prefix}_adaece"] = adaeces.item()
        self.save_results_dict(prefix, result_latex, name = f"{prefix}_corr_results{self.laplace_postfix}")


    def collect_uncertainties(self, outputs, prefix):
        ood_scores_names = self.output_ood_scores.keys()
        self.all_unc_scores = {}
        self.all_unc_scores.update(self.output_ood_scores)

        return {k:outputs[k] for k in ood_scores_names}

    def test_ood(self, collected_uncertainties, prefix, outputs = None):
        # ood_results = self.collect_uncertainties(outputs, prefix)
        ood_names = self.ood_dm.ood_datamodules_names
        result_latex = {}
        if prefix.find("test") >= 0:
            is_ind_labels = self.ood_dm.is_ind_test_labels
            is_ood_labels = self.ood_dm.is_ood_test_labels
            offsets = self.ood_dm.test_offsets
        elif prefix.find("val") >= 0:
            is_ind_labels = self.ood_dm.is_ind_val_labels
            is_ood_labels = self.ood_dm.is_ood_val_labels
            offsets = self.ood_dm.valid_offsets

        for idx,ood_labels in enumerate(is_ood_labels):
            pos_targets = torch.cat([is_ind_labels,ood_labels],dim=0)
            if outputs is not None:
                temp = pos_targets[:outputs["y_logits"].shape[0]]
                temp = temp[torch.argmax(outputs["y_logits"], dim=1) != outputs["y_true"]]-1 #remove misclassified samples from ind (do OOD and ind together)
                pos_targets[:outputs["y_logits"].shape[0]] = temp

            for k, v in collected_uncertainties.items():
                #predictions = uncertainties for IND and OOD
                print(type(v))
                predictions = torch.cat([v[0:offsets[0]], v[offsets[idx]:offsets[idx + 1]]], dim=0)
                if k.find("unc_") >= 0:

                    self.get_auroc_aupr(k, predictions, pos_targets, result_latex, prefix, ood_names[idx], self.all_unc_scores)
                    try:
                        self.plot_OOD_detection_uncertainties_histogram(k, v[0:offsets[0]],v[offsets[idx]:offsets[idx + 1]], ood_names[idx])
                    except Exception:
                        print("Error while plotting histogram")
        #creates path if does not exist
        self.save_results_dict(prefix, result_latex, name = "ood_results_new")


    def plot_OOD_detection_uncertainties_histogram(self, uncertainty_type, IND_uncertainties, OOD_uncertainties, ood_dataset_name):
        print(f"plotting histogram of {uncertainty_type} for {ood_dataset_name}")
        fig, axs = plt.subplots(nrows=1, ncols=1)

        axs.hist(IND_uncertainties.cpu().numpy(), density=True, histtype='barstacked',bins=20, alpha=0.5, label = "IND")
        axs.hist(OOD_uncertainties.cpu().numpy(), density=True, histtype='barstacked',bins=20, alpha=0.5, label = "OOD")
        axs.legend(loc='right')

        fig.savefig(self.hparams.default_root_dir + f'/{self.hparams.name}/OOD_hist_{uncertainty_type}_{ood_dataset_name}.png')
        plt.cla()




    def save_results_dict(self, prefix, result_latex, name):
        # creates path if does not exist
        logs_save_path = self.hparams.default_root_dir + "/" + self.hparams.name
        Path(logs_save_path).mkdir(parents=True, exist_ok=True)
        filename = self.get_file_name(f"{name}_{prefix}")

        with open(f'{logs_save_path}/{filename}.json', 'w') as json_file:
            json_file.write(json.dumps(result_latex, indent=4))
        ood_latex_string = ""
        for _, r in result_latex.items():
            ood_latex_string += str(r) + " & "
        with open(f'{logs_save_path}/{filename}.txt', 'w') as the_file:
            the_file.write(ood_latex_string + " \\\\")
        print("############################################################")
        print(f'SAVED IN {logs_save_path}/{filename}.json')
        print("############################################################")


    def get_auroc_aupr(self,k, predictions, pos_targets, result_latex, prefix, dataset_name, scores_to_consider):
        # compute auroc only on ood scores

        if (k in scores_to_consider and scores_to_consider[k][1] == 0):
            # if IND has low value of score, flip the labels
            print(f"for metric {k} flipping label")
            anomaly_targets = 1 - pos_targets  # IND will have low values of score, OOD will have high value
        else:
            print(f"for metric {k} NOT flipping label")
            anomaly_targets = pos_targets

        # import pdb; pdb.set_trace()
        try:
            roc_auc = roc_auc_score(anomaly_targets.cpu(), predictions.cpu())
        except Exception as e:
            print(e)
            roc_auc = -1
        try:
            prec, rec, _ = precision_recall_curve(anomaly_targets.cpu(), predictions.cpu())
            pr_auc = auc(rec, prec)

            print("MINE AUPR", str(pr_auc))
            print("SK Learn AUPR", sk.average_precision_score(anomaly_targets.cpu(), predictions.cpu()))#labels, examples))
        except Exception as e:

            print(e)
            pr_auc = -1
        self.log(f"{prefix}_{dataset_name}_{k}_AUROC", roc_auc, on_epoch=True)
        self.log(f"{prefix}_{dataset_name}_{k}_AUPR", pr_auc, on_epoch=True)

        result_latex[f"{prefix}_{dataset_name}_{k}_AUROC"] = roc_auc
        result_latex[f"{prefix}_{dataset_name}_{k}_AUPR"] = pr_auc


    def get_file_name(self, prefix = ""):
        if prefix.find("ood") >= 0:
            file_name = prefix + f"_eps{self.hparams.eps}"
        else:
            file_name = prefix

        return file_name


    def get_softmax_output(self, out, postprocess_model = None):
        if not self.output_is_logits:
            if postprocess_model is None or not hasattr(postprocess_model, "postprocess_network_outputs"):
                print("taking softmax")
                return out["y_softmax"]
            else:
                return postprocess_model.postprocess_network_outputs(out["y_softmax"])[0]

        else:
            if postprocess_model is None or not hasattr(postprocess_model, "postprocess_network_outputs"):
                return torch.softmax(out["y_logits"], dim=1)
            else:
                return torch.softmax(postprocess_model.postprocess_network_outputs(out["y_logits"])[0], dim=1)


    def get_out(self, outputs):
        if self.output_is_logits:
            aggregated = {"y_logits": [], "y_true": []}
        else:
            aggregated = {"y_softmax":[],"y_logits": [], "y_true": []}

        for i in outputs:
            for k, v in aggregated.items():
                aggregated[k].append(i[k])

        for k, v in aggregated.items():
            aggregated[k] = torch.cat(aggregated[k], dim=0)

        return aggregated

    def log_metrics(self, outputs, prefix, on_step):

        if prefix == "val":
            metrics = self.valid_metrics
        elif prefix == "test":
            metrics = self.test_metrics
        elif prefix == "train":
            metrics = self.train_metrics

        out = self.get_out(outputs)
        self.log(f"{prefix}_accuracy", metrics[f'{prefix}_accuracy'](self.get_softmax_output(out), out['y_true'].int()), prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}_ece", metrics[f'{prefix}_ece']( self.get_softmax_output(out), out['y_true'].int()),prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}_adaece", metrics[f'{prefix}_adaece']( self.get_softmax_output(out), out['y_true'].int()),prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}_nll", self.nll(out['y_logits'], out['y_true']),
                 prog_bar=True, on_step=False, on_epoch=True)

        result_latex = {}

        # if prefix == "test" or prefix == "val":
        result_latex[f"{prefix}_accuracy"] = metrics[f'{prefix}_accuracy'](self.get_softmax_output(out), out['y_true'].int()).item()
        result_latex[f"{prefix}_ece"] = metrics[f'{prefix}_ece']( self.get_softmax_output(out), out['y_true'].int()).item()
        result_latex[f"{prefix}_adaece"] = metrics[f'{prefix}_adaece']( self.get_softmax_output(out), out['y_true'].int()).item()

        self.save_results_dict(prefix, result_latex, name=f"{prefix}_acc_results")

        confs, preds = torch.max(self.get_softmax_output(out), dim=1)
        reliability_plot(confs, preds, out['y_true'].int(), self.hparams.default_root_dir + "/" + self.hparams.name + "/", num_bins=15)

    def add_uncertainty_scores(self, y_hat, embeds, result, prefix, train_gmm=None):
        result = self.add_softmaxnet_uncertainty_scores(y_hat, result,prefix)
        return result

    def add_softmaxnet_uncertainty_scores(self, y_hat, result,prefix):
        if prefix == "test" and self.evaluate_ood_mode and hasattr(self.ood_dm.ind_datamodule, "postprocess_network_outputs"):
            print("Postprocessing network outputs before uncertainty computations")
            for k, v in self.output_ood_scores.items():
                result[k] = v[0](self.ood_dm.ind_datamodule.postprocess_network_outputs(y_hat)[0], not self.output_is_logits)  # expects is_softmax
        else:
            for k,v in self.output_ood_scores.items():
                result[k] = v[0](y_hat, not self.output_is_logits) #expects is_softmax
        return result

