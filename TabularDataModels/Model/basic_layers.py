from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

# simple 2-layer dense network
class DenseThreeLayerCateg(nn.Module):

    def __init__(self,in_features, out_features, embedding_sizes, n_cont, d_rate=0.3):

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
        self.l1 = nn.Linear(self.n_emb + self.n_cont ,512)
        # define the dropout layer
        self.drop = nn.Dropout(d_rate)
        # define a second FCN
        self.l2 = nn.Linear(512,256)
        # define the second dropout layer
        self.l3 = nn.Linear(256,128)
        self.drop = nn.Dropout(d_rate)
        # dropout for embedding
        self.emb_drop = nn.Dropout(0.5)
        # define the final prediction layer
        self.ol = nn.Linear(128,self.o_dim)
        # batch norm layers, guideline for batchnorm: https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x_cont, x_cat):

        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x = torch.cat([x, x_cont], 1)
        x = self.l1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.l2(x)
        x = F.relu(self.bn2(x))
        x = self.drop(x)
        x = self.l3(x)
        x = F.relu(self.bn3(x))
        x = self.drop(x)
        return self.ol(x)

    # def forward(self, x_cont, x_cat):

    #     x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
    #     x = torch.cat(x, 1)
    #     x = self.emb_drop(x)
    #     x = torch.cat([x, x_cont], 1)
    #     x = F.relu(self.l1(x))
    #     x = self.drop(x)
    #     x = F.relu(self.l2(x))
    #     x = self.drop(x)
    #     x = F.relu(self.l3(x))
    #     x = self.drop(x)
    #     return self.ol(x)

# simple 2-layer dense network with categorical embedding
class DenseThreeLayer(nn.Module):

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
    
class ConvThreeLayerCateg(nn.Module):
    def __init__(self, in_features, out_features, embedding_sizes, n_cont, d_rate=0.3):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)
        self.n_emb, self.n_cont = n_emb, n_cont
        self.o_dim = out_features
        self.d_rate = d_rate
        self.conv1 = nn.Conv1d(in_channels=self.n_emb + self.n_cont, out_channels=512, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
        self.drop = nn.Dropout(d_rate)
        self.emb_drop = nn.Dropout(0.5)
        self.ol = nn.Linear(128, self.o_dim)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
    def forward(self, x_cont, x_cat):
        x = [e(x_cat[:,i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        # combine embeddings and continuous variables
        x = torch.cat([x.unsqueeze(2), x_cont.unsqueeze(2)], 1)
        # permute the tensor to [batch_size, <total_embeddings_output>+1, 1]
        # x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.drop(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.drop(x)
        x = x.squeeze() # swap the last two dimensions back to the original order
        x = self.ol(x)
        return x
