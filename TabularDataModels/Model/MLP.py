from .BaseModel import BaseLightningModule
from .basic_layers import DenseThreeLayer, DenseThreeLayerCateg, MaskedDenseThreeLayerCateg
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch


class Mlp(BaseLightningModule):
    
    def __init__(self, *args, **kwargs):
        super(Mlp, self).__init__(*args, **kwargs)

    # def create_model(self, n_cont, out_features, cat_features, emb_size):
    #     if cat_features:
    #         return DenseThreeLayerCateg((n_cont, self.n_categ), out_features, emb_size, n_cont)
    #     else:
    #         return DenseThreeLayer(n_cont, out_features)
        
    def create_model(self, n_cont, out_features, cat_features, emb_size):
        if cat_features:
            return MaskedDenseThreeLayerCateg((n_cont, self.n_categ), out_features, emb_size, n_cont)
        else:
            return MaskedDenseThreeLayerCateg(n_cont, out_features)
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
        return {"optimizer":optimizer, "lr_scheduler":scheduler, "monitor":"val_loss"}