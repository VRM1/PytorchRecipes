import pytorch_lightning as pl
import torch
import torchmetrics
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from Utils import BatchDataLoader, DataRepo
from tqdm.auto import tqdm
from Utils.CustomMetrics import FocalLoss
import torch.nn as nn



class BaseLightningModule(pl.LightningModule):
    
    def __init__(self, epochs, in_features, out_features, batch_size,
                  workers=1, learning_rate=0.001, emb_size=None,
                    cat_features=False, **kwargs):
        super(BaseLightningModule, self).__init__()
        
        task_typ = 'binary'
        self.epochs = epochs
        self.batch_size = batch_size
        self.workers = workers
        self.n_cont, self.n_categ = in_features
        self.emb_size = emb_size
        self.lr = learning_rate
        self.args = kwargs.get('args')
        num_class = 1

        if out_features > 2:
            task_typ = 'multiclass'

        # Model-specific initialization
        self.model = self.create_model(n_cont=self.n_cont, out_features=out_features,
                                        cat_features=cat_features, emb_size=emb_size)
        self.configure_optimizers()

        self.t_outputs = []
        self.v_outputs = []
        self.te_outputs = []
        self.t_len = 0

        self.acc = torchmetrics.Accuracy(num_classes=num_class, task=task_typ)
        self.auc_roc = torchmetrics.AUROC(num_classes=num_class, task=task_typ)
        self.auc_prec = torchmetrics.AveragePrecision(num_classes=num_class, task=task_typ)

        self.save_hyperparameters()
        print(self.hparams)

    def forward(self, batch):
        if len(batch) > 2:
            # we have a categorical feature
            x_num, x_cat, y = batch
            x_num = x_num.view(x_num.size(0), -1)
            x_cat = x_cat.view(x_cat.size(0), -1)
            y = y.flatten()
            out = self.model(x_num, x_cat)
        else:
            x_num, y = batch
            x_num = x_num.view(x_num.size(0), -1)
            out = self.model(x_num)
        if len(x_num.shape) == 3:
            # implemented to handle lazy data loader (not currently functional)
            x_num = x_num.view(-1,x_num.shape[2])

        return (out, y)
    
    def _prepare_dataloader(self, mode='train'):
        data_repo = DataRepo()
        fl_loader = data_repo(self.args)
        fl_loader.setup(mode)
        if mode == 'train':
            fl_loader = fl_loader.train_dataloader()
        elif mode == 'validate':
            fl_loader = fl_loader.val_dataloader()
        elif mode == 'test':
            fl_loader = fl_loader.predict_dataloader()
        
        splits = self.__get_batch_size(fl_loader)
        
        def generator():
            for epoch in range(self.epochs):
                for i, d in tqdm(enumerate(fl_loader)):
                    data_loader = DataLoader(BatchDataLoader(d), batch_size=self.batch_size,
                                                num_workers=self.workers, pin_memory=True,
                                                prefetch_factor=64)
                    for batch in data_loader:
                        if mode == 'train':
                            self.t_len += batch[0].shape[0]
                        yield batch
                        
        progress_bar = tqdm(
            generator(),
            total=len(fl_loader) * splits,
            desc=mode.capitalize()
        )
        return progress_bar
    
    def __get_batch_size(self, file_loader):
        d = next(enumerate(file_loader))[1]
        data_loader = DataLoader(BatchDataLoader(d), batch_size=self.batch_size)
        return len(data_loader)
    
    def create_model(self, n_cont, out_features, cat_features, emb_size):
        """To be overridden by derived classes"""
        raise NotImplementedError
    
    def configure_optimizers(self):
        """To be overridden by derived classes"""
        raise NotImplementedError
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
       return self._prepare_dataloader('train')

    def val_dataloader(self) -> EVAL_DATALOADERS:
    
       return self._prepare_dataloader('validate')
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        
        return self._prepare_dataloader('test')
    
    def l1_penalty(self, var):
        return torch.abs(var).sum()
    
    def compute_loss_and_metrics(self, batch):
        lambda1 = 0.0001
        out, y = self(batch)
        loss_fn = nn.CrossEntropyLoss()
        # loss_fn = FocalLoss()
        loss = loss_fn(out, y)
        loss += lambda1 * (self.l1_penalty(self.model.mask_emb) + self.l1_penalty(self.model.mask_cont))
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
        
    def validation_step(self, batch, batch_idx):

        y, loss, prob_ones, accuracy = self.compute_loss_and_metrics(batch)
        auc_precision = self.auc_prec(prob_ones, y)
        auc_roc = self.auc_roc(prob_ones, y)
        metrics = {
            'val_loss': loss, 
            'accuracy': accuracy,
            'auc_prec': auc_precision, 
            'auc_roc': auc_roc
        }
        self.v_outputs.append(metrics)
        return metrics

    def predict_step(self, batch, batch_idx):

        return self(batch)

    def on_training_epoch_end(self) -> None:
        
        loss = sum(output['loss'] for output in self.t_outputs) / len(self.t_outputs)
        acc = sum(output['accuracy'] for output in self.t_outputs) / len(self.t_outputs)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.t_outputs.clear() # free memory

    
    def on_validation_epoch_end(self) -> None:
        
        print(f'Total Train Len:{self.t_len}')
        loss = sum(output['val_loss'] for output in self.v_outputs) / len(self.v_outputs)
        acc = sum(output['accuracy'] for output in self.v_outputs) / len(self.v_outputs)
        avg_auc_prec = sum(output['auc_prec'] for output in self.v_outputs) / len(self.v_outputs)
        avg_auc_roc = sum(output['auc_roc'] for output in self.v_outputs) / len(self.v_outputs)
        self.log("val_loss", loss)
        self.log("valid_acc", acc)
        self.log("valid_auc_prec", avg_auc_prec)
        self.log("valid_auc_roc", avg_auc_roc)
        self.t_len = 0
        self.v_outputs.clear() # free memory
    
    def on_test_epoch_end(self) -> None:

        loss = sum(output['test_loss'] for output in self.te_outputs) / len(self.te_outputs)
        acc = sum(output['test_accuracy'] for output in self.te_outputs) / len(self.te_outputs)
        auc_prec = sum(output['test_auc_prec'] for output in self.te_outputs) / len(self.te_outputs)
        avg_auc_roc = sum(output['test_auc_roc'] for output in self.te_outputs) / len(self.te_outputs)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.log("test_auc_prec", auc_prec)
        self.log("test_auc_roc", avg_auc_roc)
        self.te_outputs.clear() # free memory