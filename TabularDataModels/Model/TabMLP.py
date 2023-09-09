import torch
import pytorch_lightning as pl
from pytorch_widedeep.models import TabMlp
from pytorch_widedeep.metrics import Accuracy
import torchmetrics
import torch.nn as nn
from .Basemodel import BaseLightningModule

class TabMLP(BaseLightningModule):
    
    def __init__(self, column_indx, num_columns, **kwargs):
        
        self.column_indx = column_indx
        self.num_clms = num_columns
        super(TabMLP, self).__init__(**kwargs)

    def create_model(self, **kwargs):
        cat_features = kwargs.get('cat_features')
        if cat_features:
            #Define the FTTransformer model
            model = TabMlp(
                column_idx=self.column_indx,
                cat_embed_input=self.emb_size,
                continuous_cols=self.num_clms,
                mlp_hidden_dims=[512, 256, 128],
                mlp_dropout=0.2,
                mlp_activation="relu",
                embed_continuous=True,
                mlp_batchnorm=True
            )
        else:
            model = TabMlp(
                column_idx=self.column_indx,
                continuous_cols=self.num_clms,
                mlp_hidden_dims=[512, 256, 128],
                mlp_dropout=0.5,
                mlp_activation="leaky_relu",
                embed_continuous=True,
                mlp_batchnorm=True
            )
        return model
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def forward(self, batch):
        if len(batch) > 2:
            # we have a categorical feature
            x_num, x_cat, y = batch
            x = torch.cat([x_num, x_cat], axis=1)
            x_num = x_num.view(x_num.size(0), -1)
            x_cat = x_cat.view(x_cat.size(0), -1)
            y = y.flatten()
            out = self.model(x)
        else:
            x_num, y = batch
            x_num = x_num.view(x_num.size(0), -1)
            out = self.model(x_num)

        return (out, y)
