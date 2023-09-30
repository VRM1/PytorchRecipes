import torch
import pytorch_lightning as pl
from pytorch_widedeep.models import SelfAttentionMLP
from pytorch_widedeep.metrics import Accuracy
import torchmetrics
import torch.nn as nn
from .BaseBatchModel import BaseLightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

class AttentionMLP(BaseLightningModule):
    
    def __init__(self, column_indx, num_columns, **kwargs):
        
        self.column_indx = column_indx
        self.num_clms = num_columns
        super(AttentionMLP, self).__init__(**kwargs)

    def create_model(self, **kwargs):
        cat_features = kwargs.get('cat_features')
        if cat_features:
            #Define the FTTransformer model
            model = SelfAttentionMLP(
                column_idx=self.column_indx,
                cat_embed_input=self.emb_size,
                continuous_cols=self.num_clms,
                cat_embed_activation="relu",
                cont_embed_activation="relu",
            )
        else:
            model = SelfAttentionMLP(
                column_idx=self.column_indx,
                continuous_cols=self.num_clms,
                cont_embed_activation="relu",
            )
        self.ol = nn.Linear(640, 2)
        return model
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, verbose=True)
        return {"optimizer":optimizer, "lr_scheduler":scheduler, "monitor":"val_loss"}
    
    def forward(self, batch):
        if len(batch) > 2:
            # we have a categorical feature
            x_num, x_cat, y = batch
            x = torch.cat([x_num, x_cat], axis=1)
            x_num = x_num.view(x_num.size(0), -1)
            x_cat = x_cat.view(x_cat.size(0), -1)
            y = y.flatten()
            out = self.model(x)
            out = self.ol(out)
        else:
            x_num, y = batch
            x_num = x_num.view(x_num.size(0), -1)
            out = self.model(x_num)
            out = self.ol(out)

        return (out, y)
