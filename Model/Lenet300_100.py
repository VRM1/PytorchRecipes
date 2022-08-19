import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

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
        # define a second FCN
        self.l2 = nn.Linear(300,100)
        # define the final prediction layer
        self.ol = nn.Linear(100,self.o_dim)

    def forward(self,x):

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.ol(x)


class SimpleLenet(pl.LightningModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.dense = Dense2(in_features, out_features)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        out = self.dense(x)
        loss = nn.CrossEntropyLoss()
        loss = loss(out, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# class Lenet300_100(nn.Module):
#
#     def __init__(self,in_features, out_features,d_rate=0.2):
#
#         super().__init__()
#         # input dimension
#         self.i_dim = in_features
#         # output dimension
#         self.o_dim = out_features
#         # dropout rate
#         self.d_rate = d_rate
#         # define a fully connected layer
#         self.l1 = nn.Linear(self.i_dim ,300)
#         # define the dropout layer
#         self.l1_drop = nn.Dropout(d_rate)
#         # define a second FCN
#         self.l2 = nn.Linear(300,100)
#         # define the second dropout layer
#         self.l2_drop = nn.Dropout(d_rate)
#         # define the final prediction layer
#         self.ol = nn.Linear(100,self.o_dim)
#
#     def forward(self,x):
#
#         x = F.relu(self.l1(x))
#         x = self.l1_drop(x)
#         x = F.relu(self.l2(x))
#         x = self.l2_drop(x)
#         return self.ol(x)
