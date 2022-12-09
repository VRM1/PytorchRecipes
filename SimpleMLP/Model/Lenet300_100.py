from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

# simple 2-layer dense network
class Dense2(nn.Module):

    def __init__(self,in_features, out_features,d_rate=0.2):

        super().__init__()
        # input dimension
        self.i_dim = in_features
        # output dimension
        self.o_dim = out_features
        # dropout rate
        self.d_rate = d_rate
        # define a fully connected layer
        self.l1 = nn.Linear(self.i_dim ,300)
        # define the dropout layer
        self.l1_drop = nn.Dropout(d_rate)
        # define a second FCN
        self.l2 = nn.Linear(300,100)
        # define the second dropout layer
        self.l2_drop = nn.Dropout(d_rate)
        # define the final prediction layer
        self.ol = nn.Linear(100,self.o_dim)

    def forward(self,x):

        x = F.relu(self.l1(x))
        x = self.l1_drop(x)
        x = F.relu(self.l2(x))
        x = self.l2_drop(x)
        return self.ol(x)


class SimpleLenet(pl.LightningModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        task_typ = 'binary'
        if out_features > 2:
            task_typ = 'multiclass'
        self.dense = Dense2(in_features, out_features)
        self.acc = torchmetrics.Accuracy(task=task_typ)
        self.auc_roc = torchmetrics.AUROC(num_classes=2)
        self.auc_prec = torchmetrics.AveragePrecision(pos_label=1)
                # task type of calculating accuracy
        
    def forward(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        out = self.dense(x)
        return (out, y)

    def predict_step(self, batch, batch_idx):

        return self(batch)


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        out = self.dense(x)
        loss = nn.CrossEntropyLoss()
        loss = loss(out, y)
        accuracy = self.acc(out.softmax(dim=-1), y)
        return {'loss':loss, 'accuracy':accuracy}
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        out = self.dense(x)
        loss = nn.CrossEntropyLoss()
        preds = out.softmax(dim=-1)
        loss = loss(out, y)
        accuracy = self.acc(preds, y)
        auc_precision = self.auc_prec(preds[:,1], y)
        auc_roc = self.auc_roc(preds, y)
        return {'test_loss':loss, 'test_accuracy':accuracy, \
             'test_auc_prec':auc_precision, 'test_auc_roc':auc_roc}
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        out = self.dense(x)
        loss = nn.CrossEntropyLoss()
        loss = loss(out, y)
        preds = out.softmax(dim=-1)
        accuracy = self.acc(preds, y)
        auc_precision = self.auc_prec(preds[:,1], y)
        auc_roc = self.auc_roc(preds, y)
        return {'val_loss':loss, 'accuracy':accuracy, \
             'auc_prec':auc_precision, 'auc_roc':auc_roc}
        # return loss
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    # def training_epoch_end(self, outputs) -> None:
    #     loss = sum(output['loss'] for output in outputs) / len(outputs)
    #     acc = sum(output['accuracy'] for output in outputs) / len(outputs)
    #     print ({'train_loss:':loss.item(), 'train_acc:':acc.item()})
    
    def validation_epoch_end(self, outputs):
        loss = sum(output['val_loss'] for output in outputs) / len(outputs)
        acc = sum(output['accuracy'] for output in outputs) / len(outputs)
        avg_auc_prec = sum(output['auc_prec'] for output in outputs) / len(outputs)
        avg_auc_roc = sum(output['auc_roc'] for output in outputs) / len(outputs)
        self.log("val_loss", loss)
        self.log("valid_acc", acc)
        self.log("auc_prec", avg_auc_prec)
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

