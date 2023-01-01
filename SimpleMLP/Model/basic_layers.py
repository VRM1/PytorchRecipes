from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

# simple 2-layer dense network
class DenseTwoLayerCateg(nn.Module):

    def __init__(self,in_features, out_features, embedding_sizes, n_cont, d_rate=0.2):

        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) \
             for categories,size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)
        self.n_emb, self.n_cont = n_emb, n_cont

        # output dimension
        self.o_dim = out_features
        # dropout rate
        self.d_rate = d_rate
        # define a fully connected layer
        self.l1 = nn.Linear(self.n_emb + self.n_cont ,300)
        # define the dropout layer
        self.l1_drop = nn.Dropout(d_rate)
        # define a second FCN
        self.l2 = nn.Linear(300,100)
        # define the second dropout layer
        self.l2_drop = nn.Dropout(d_rate)
        # dropout for embedding
        self.emb_drop = nn.Dropout(d_rate)
        # define the final prediction layer
        self.ol = nn.Linear(100,self.o_dim)

    def forward(self, x_cont, x_cat):

        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x = torch.cat([x, x_cont], 1)
        x = F.relu(self.l1(x))
        x = self.l1_drop(x)
        x = F.relu(self.l2(x))
        x = self.l2_drop(x)
        return self.ol(x)

# simple 2-layer dense network with categorical embedding
class DenseTwoLayer(nn.Module):

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