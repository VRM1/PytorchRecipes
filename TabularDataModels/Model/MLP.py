from .Basemodel import BaseLightningModule
from .basic_layers import DenseThreeLayer, DenseThreeLayerCateg
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

# from Utils import CustomDataLoader
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from tqdm.auto import tqdm

class Mlp(BaseLightningModule):
    
    def __init__(self, *args, **kwargs):
        super(Mlp, self).__init__(*args, **kwargs)

    def create_model(self, n_cont, out_features, cat_features, emb_size):
        if cat_features:
            return DenseThreeLayerCateg((n_cont, self.n_categ), out_features, emb_size, n_cont)
        else:
            return DenseThreeLayer(n_cont, out_features)
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #     return optimizer
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
        return {"optimizer":optimizer, "lr_scheduler":scheduler, "monitor":"val_loss"}