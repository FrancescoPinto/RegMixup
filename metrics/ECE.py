from torchmetrics import Metric
import torch
import torch.nn.functional as F

class ECE(Metric):
    '''
    Compute ECE (Expected Calibration Error)
    '''

    def __init__(self, n_bins=15, process_logits=True, dist_sync_on_step=False):
        super(ECE, self).__init__(dist_sync_on_step=dist_sync_on_step)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        for _, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            self.add_state(f"accuracies_{_}",default=torch.tensor(0.), dist_reduce_fx="sum")
            self.add_state(f"confidences_{_}", default=torch.tensor(0.), dist_reduce_fx="sum")
            self.add_state(f"total_{_}", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state(f"total", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.process_logits = process_logits

    def update(self, input, labels):
        if self.process_logits:
            softmax = F.softmax(input, dim=1)
            confidences, predictions = torch.max(softmax, 1)
        else:
            confidences, predictions = torch.max(input, 1)

        accuracies = predictions.eq(labels)

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