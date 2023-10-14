from .BaseModel import BaseLightningModule
from .basic_layers import BayesianMLP
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn

class Mlp(BaseLightningModule):
    
    def __init__(self, *args, **kwargs):
        super(Mlp, self).__init__(*args, **kwargs)

    def create_model(self, n_cont, out_features, cat_features, emb_size):
        if cat_features:
            return BayesianMLP((n_cont, self.n_categ), out_features, emb_size, n_cont)
        else:
            return BayesianMLP(n_cont, out_features)
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
        return {"optimizer":optimizer, "lr_scheduler":scheduler, "monitor":"val_loss"}
    
    def compute_loss_and_metrics(self, batch):
        out, y = self(batch)
        kl = self.model.kl_divergence()
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(out, y)
        preds = out.softmax(dim=-1)
        prob_ones = preds[:,1]
        accuracy = self.acc(prob_ones, y)
        
        return y, loss, prob_ones, accuracy

    def training_step(self, batch, batch_idx):

        y, loss, prob_ones, accuracy = self.compute_loss_and_metrics(batch)
        metrics = {
            'global_step': self.current_epoch,
            'loss': loss,
            'accuracy': accuracy
        }
        self.t_outputs.append(metrics)
        return metrics