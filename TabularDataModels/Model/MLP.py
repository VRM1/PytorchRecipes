import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from .basic_layers import DenseThreeLayer, DenseThreeLayerCateg
from Utils import CustomDataLoader
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.auto import tqdm
import math
import pdb
class Mlp(pl.LightningModule):
    def __init__(self, epochs, in_features, out_features, file_loader,
                 batch_size, workers=1, emb_size=None, cat_features=False):
        super().__init__()
        task_typ = 'binary'
        self.epochs = epochs
        self.file_loader = file_loader
        self.batch_size = batch_size
        self.workers = workers
        num_class = 1
        n_cont, n_categ = in_features
        if out_features > 2:
            task_typ = 'multiclass'
        if cat_features:
            self.dense = DenseThreeLayerCateg(in_features, out_features, emb_size, n_cont)
        else:
            self.dense = DenseThreeLayer(n_cont, out_features)
        self.t_outputs = []
        self.v_outputs = []
        self.te_outputs = []
        self.t_len = 0
        self.acc = torchmetrics.Accuracy(num_classes=num_class, task=task_typ)
        self.auc_roc = torchmetrics.AUROC(num_classes=num_class, task=task_typ)
        self.auc_prec = torchmetrics.AveragePrecision(num_classes=num_class, task=task_typ)
        self.save_hyperparameters()

    def forward(self, batch):
        if len(batch) > 2:
            # we have a categorical feature
            x_num, x_cat, y = batch
            x_num = x_num.view(x_num.size(0), -1)
            x_cat = x_cat.view(x_cat.size(0), -1)
            y = y.flatten()
            out = self.dense(x_num, x_cat)
        else:
            x_num, y = batch
            x_num = x_num.view(x_num.size(0), -1)
            out = self.dense(x_num)
        if len(x_num.shape) == 3:
            # implemented to handle lazy data loader (not currently functional)
            x_num = x_num.view(-1,x_num.shape[2])

        return (out, y)

    def predict_step(self, batch, batch_idx):

        return self(batch)

    def training_step(self, batch, batch_idx):
        out, y = self(batch)
        loss = nn.CrossEntropyLoss()
        loss = loss(out, y)
        preds = out.softmax(dim=-1)
        prob_ones = preds[:,1]
        accuracy = self.acc(prob_ones, y)
        # log the loss to the progress bar and the logger
        metrics = {'global_step':self.current_epoch, 'loss':loss, \
                    'accuracy':accuracy}
        self.t_outputs.append(metrics)
        return metrics
    
    def validation_step(self, batch, batch_idx):
        out, y = self(batch)
        loss = nn.CrossEntropyLoss()
        loss = loss(out, y)
        preds = out.softmax(dim=-1)
        prob_ones = preds[:,1]
        accuracy = self.acc(prob_ones, y)
        auc_precision = self.auc_prec(prob_ones, y)
        auc_roc = self.auc_roc(prob_ones, y)
        metrics = {'val_loss':loss, 'accuracy':accuracy, \
                'auc_prec':auc_precision, 'auc_roc':auc_roc}
        self.v_outputs.append(metrics)
        return metrics
    
    def __get_batch_size(self, file_loader):
            d = next(enumerate(file_loader))[1]
            data_loader = DataLoader(CustomDataLoader(d), batch_size=self.batch_size)
            return len(data_loader)
            
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        self.file_loader.setup('fit')
        # return self.file_loader.train_dataloader()
        tr_file_loader = self.file_loader.train_dataloader()
        splits = self.__get_batch_size(tr_file_loader)
        def generator():
            for epoch in range(self.epochs):
                for i, d in tqdm(enumerate(tr_file_loader)):
                    data_loader = DataLoader(CustomDataLoader(d), batch_size=self.batch_size,
                                                num_workers=self.workers, pin_memory=True,
                                                  prefetch_factor=64)
                    for batch in data_loader:
                        self.t_len += batch[0].shape[0]
                        yield batch
        
        progress_bar = tqdm(
            generator(),
            total=len(tr_file_loader) * splits,
            desc="Training"
        )
        return progress_bar

    def val_dataloader(self) -> EVAL_DATALOADERS:
        
        self.file_loader.setup('fit')
        # return self.file_loader.val_dataloader()

        val_file_loader = self.file_loader.val_dataloader()
        splits = self.__get_batch_size(val_file_loader)
        def generator():
            for epoch in range(self.epochs):
                for i, d in tqdm(enumerate(val_file_loader)):
                    data_loader = DataLoader(CustomDataLoader(d), batch_size=self.batch_size,
                                                num_workers=self.workers, pin_memory=True,
                                                  prefetch_factor=64)
                    for new_data in data_loader:
                        yield new_data
        progress_bar = tqdm(
            generator(),
            total=len(val_file_loader) * splits,
            desc='Validating'
        )
        return progress_bar
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        
        self.file_loader.setup('test')
        t_file_loader = self.file_loader.predict_dataloader()
        splits = self.__get_batch_size(t_file_loader)
        def generator():
            for epoch in range(self.epochs):
                for i, d in tqdm(enumerate(t_file_loader)):
                    data_loader = DataLoader(CustomDataLoader(d), batch_size=self.batch_size,
                                              num_workers=self.workers, pin_memory=True,
                                                prefetch_factor=64)
                    for new_data in data_loader:
                        yield new_data
        progress_bar = tqdm(
            generator(),
            total=len(t_file_loader) * splits,
            desc='Testing'
        )
        return progress_bar
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
    
        self.file_loader.setup('test')
        t_file_loader = self.file_loader.predict_dataloader()
        splits = self.__get_batch_size(t_file_loader)
        def generator():
            for epoch in range(self.epochs):
                for i, d in tqdm(enumerate(t_file_loader)):
                    data_loader = DataLoader(CustomDataLoader(d), batch_size=self.batch_size,
                                            num_workers=self.workers, pin_memory=True,
                                                prefetch_factor=64)
                    for new_data in data_loader:
                        yield new_data
        progress_bar = tqdm(
            generator(),
            total=len(t_file_loader) * splits,
            desc='Testing'
        )
        return progress_bar
    
    def test_step(self, batch, batch_idx):
        out, y = self(batch)
        loss = nn.CrossEntropyLoss()
        preds = out.softmax(dim=-1)
        # get the probability of predicting +ve class
        prob_ones = preds[:,1]
        loss = loss(out, y)
        self.test_step_outputs.append(loss)
        accuracy = self.acc(prob_ones, y)
        auc_precision = self.auc_prec(prob_ones, y)
        auc_roc = self.auc_roc(prob_ones, y)
        metric = {'test_loss':loss, 'test_accuracy':accuracy, \
             'test_auc_prec':auc_precision, 'test_auc_roc':auc_roc}
        self.te_outputs.append(metric)
        return metric
    


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
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
