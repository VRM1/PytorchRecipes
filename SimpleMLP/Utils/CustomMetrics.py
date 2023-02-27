from torchmetrics import Metric, ROC
import torch
from scipy.interpolate import interp1d

class FprRatio(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("num_p", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_n", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fpr", default=torch.tensor(0))
        self.add_state("tpr", default=torch.tensor(0))

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = self._input_format(preds, target) # needs to be implemented
        assert preds.shape == target.shape
        roc = ROC(task='binary')
        self.fpr, self.tpr, thresholds = roc(preds, target)
        self.num_p = torch.sum(target)
        self.num_n = target.shape[0] - self.num_p

    def compute(self):
        fp_ratio = self.fpr * self.num_n/(self.tpr * self.num_p + 0.0001)
        # this is just a dummy example on how a custom metric is designed
        return fp_ratio[10]