from torchmetrics import Metric, ROC
import torch
import torch.nn as nn
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
    

# class FocalLoss(Metric):
#     def __init__(self, alpha=1, gamma=2, reduce=True, dist_sync_on_step=False):
#         super(FocalLoss, self).__init__(dist_sync_on_step=dist_sync_on_step)
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduce = reduce

#         # Internal metric state
#         self.add_state("total_loss", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
#         self.add_state("num_samples", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")

#     def update(self, preds, target):
#         # Ensure the state tensors are on the same device as the inputs
#         self.total_loss = self.total_loss.to(preds.device)
#         self.num_samples = self.num_samples.to(preds.device)

#         BCE_loss = nn.CrossEntropyLoss(reduce=False)(preds, target)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

#         if self.reduce:
#             self.total_loss += torch.sum(F_loss)
#         else:
#             self.total_loss += F_loss

#         self.num_samples += target.size(0)

#     def compute(self):
#         return self.total_loss / self.num_samples


class FocalLoss(Metric):
    def __init__(self, alpha_pos=0.6, alpha_neg=0.4, gamma=1.5, reduce=True, dist_sync_on_step=False):
        super(FocalLoss, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma = gamma
        self.reduce = reduce

        # Internal metric state
        self.add_state("total_loss", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds, target):
        # Ensure the state tensors are on the same device as the inputs
        self.total_loss = self.total_loss.to(preds.device)
        self.num_samples = self.num_samples.to(preds.device)

        BCE_loss = nn.CrossEntropyLoss(reduce=False)(preds, target)
        pt = torch.exp(-BCE_loss)
        
        # Incorporating alpha balancing
        alpha_t = torch.where(target == 1, self.alpha_pos, self.alpha_neg).to(preds.device)
        
        F_loss = alpha_t * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            self.total_loss += torch.sum(F_loss)

        self.num_samples += target.size(0)

    def compute(self):
        return self.total_loss / self.num_samples
