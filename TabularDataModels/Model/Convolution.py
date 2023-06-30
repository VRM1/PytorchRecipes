import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from .basic_layers import ConvThreeLayerCateg, SoftOrdering1DCNN
from Utils import FprRatio

class Conv(pl.LightningModule):
    def __init__(self, in_features, out_features, \
        emb_size=None, cat_features=False):
        super().__init__()
        
        task_typ = 'binary'
        num_class = 1
        n_cont, n_categ = in_features
        if out_features > 2:
            task_typ = 'multiclass'
        if cat_features:
            self.model = ConvThreeLayerCateg(in_features, out_features, emb_size, n_cont)

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
            out = self.model(x_num, x_cat)
        else:
            x_num, y = batch
            x_num = x_num.view(x_num.size(0), -1)
            out = self.model(x_num)
        if len(x_num.shape) == 3:
            # implemented to handle lazy data loader (not currently functional)
            x_num = x_num.view(-1,x_num.shape[2])

        return (out, y)

    def predict_step(self, batch, batch_idx):

        return self(batch)


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
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
        loss = nn.CrossEntropyLoss()
        loss = loss(out, y)
        preds = out.softmax(dim=-1)
        prob_ones = preds[:,1]
        accuracy = self.acc(prob_ones, y)
        return {'loss':loss, 'accuracy':accuracy}
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
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

        loss = nn.CrossEntropyLoss()
        preds = out.softmax(dim=-1)
        # get the probability of predicting +ve class
        prob_ones = preds[:,1]
        loss = loss(out, y)
        accuracy = self.acc(prob_ones, y)
        auc_precision = self.auc_prec(prob_ones, y)
        auc_roc = self.auc_roc(prob_ones, y)
        return {'test_loss':loss, 'test_accuracy':accuracy, \
             'test_auc_prec':auc_precision, 'test_auc_roc':auc_roc}
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
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
            
        loss = nn.CrossEntropyLoss()
        loss = loss(out, y)
        preds = out.softmax(dim=-1)
        prob_ones = preds[:,1]
        accuracy = self.acc(prob_ones, y)
        auc_precision = self.auc_prec(prob_ones, y)
        auc_roc = self.auc_roc(prob_ones, y)
        return {'val_loss':loss, 'accuracy':accuracy, \
             'auc_prec':auc_precision, 'auc_roc':auc_roc}
        # return loss
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

    def validation_epoch_end(self, outputs):
        loss = sum(output['val_loss'] for output in outputs) / len(outputs)
        acc = sum(output['accuracy'] for output in outputs) / len(outputs)
        avg_auc_prec = sum(output['auc_prec'] for output in outputs) / len(outputs)
        avg_auc_roc = sum(output['auc_roc'] for output in outputs) / len(outputs)
        self.log("val_loss", loss)
        self.log("valid_acc", acc)
        self.log("valid_auc_prec", avg_auc_prec)
        self.log("valid_auc_roc", avg_auc_roc)
    
    def test_epoch_end(self, outputs):
        loss = sum(output['test_loss'] for output in outputs) / len(outputs)
        acc = sum(output['test_accuracy'] for output in outputs) / len(outputs)
        auc_prec = sum(output['test_auc_prec'] for output in outputs) / len(outputs)
        avg_auc_roc = sum(output['test_auc_roc'] for output in outputs) / len(outputs)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.log("test_auc_prec", auc_prec)
        self.log("test_auc_roc", avg_auc_roc)

