from torchmetrics import Metric
import torch
import torch.nn.functional as F
import numpy as np

class AdaECE(Metric):
    '''
    Compute Adaptive ECE (Adaptive Expected Calibration Error). WARNING: for this metric to work appropriately the update method must be called only once on the whole dataset (due to the dynamic binning)
    '''

    def __init__(self, n_bins=15, process_logits=True, dist_sync_on_step=False):
        super(AdaECE, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.process_logits = process_logits
        self.nbins = n_bins


    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))


    def update(self, input, labels):
        if self.process_logits:
            softmax = F.softmax(input, dim=1)
            confidences, predictions = torch.max(softmax, 1)
        else:
            confidences, predictions = torch.max(input, 1)

        accuracies = predictions.eq(labels)
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

        for _, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            self.add_state(f"accuracies_{_}",default=torch.tensor(0.), dist_reduce_fx="sum")
            self.add_state(f"confidences_{_}", default=torch.tensor(0.), dist_reduce_fx="sum")
            self.add_state(f"total_{_}", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state(f"total", default=torch.tensor(0.), dist_reduce_fx="sum")


        self.total += labels.numel()

        for _, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            #prop_in_bin = in_bin.float().mean() #this is equivalent to nb/N
            total_for_bin = in_bin.float().sum()
            if  total_for_bin > 0:
                setattr(self, f"accuracies_{_}", getattr(self, f"accuracies_{_}") + accuracies[in_bin].float().sum())
                setattr(self, f"confidences_{_}", getattr(self, f"confidences_{_}") + confidences[in_bin].sum())
                setattr(self, f"total_{_}", getattr(self, f"total_{_}") + total_for_bin)

    def compute(self):
        ece = 0
        for _, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            if getattr(self, f"total_{_}").item() > 0:
                avg_confidence_in_bin = getattr(self, f"confidences_{_}")/ getattr(self, f"total_{_}")
                avg_accuracy_in_bin = getattr(self, f"accuracies_{_}")/ getattr(self, f"total_{_}")
                prop_in_bin = getattr(self, f"total_{_}")/ getattr(self, f"total")
                ece += torch.abs(avg_confidence_in_bin - avg_accuracy_in_bin) * prop_in_bin

        return ece