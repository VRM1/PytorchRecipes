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
        dense1 = nn.Linear(self.n_emb + self.n_cont, 1024, bias=False)
        self.dense1 = nn.utils.weight_norm(dense1)
        # self.conv1 = nn.Conv1d(in_channels=self.n_emb + self.n_cont, out_channels=512, kernel_size=1)
        self.conv1 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
        self.drop = nn.Dropout(d_rate)
        self.emb_drop = nn.Dropout(0.5)
        self.ol = nn.Linear(512, self.o_dim)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.flt = nn.Flatten()
        
    def forward(self, x_cont, x_cat):
        x = [e(x_cat[:,i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        # combine embeddings and continuous variables
        # x = torch.cat([x.unsqueeze(2), x_cont.unsqueeze(2)], 1)
        x = torch.cat([x, x_cont], 1)
        x = nn.functional.relu(self.dense1(x))
        x = x.reshape(x.shape[0], 256, -1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.drop(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.flt(x)
        x = self.drop(x)
        # x = x.squeeze() # swap the last two dimensions back to the original order
        x = self.ol(x)
        return x

# https://medium.com/spikelab/convolutional-neural-networks-on-tabular-datasets-part-1-4abdd67795b6
class SoftOrdering1DCatCNN(pl.LightningModule):

    def __init__(self, in_features, out_features, embedding_sizes, n_cont, sign_size=32, cha_input=16, cha_hidden=32, 
                 K=2, dropout_input=0.2, dropout_hidden=0.2, dropout_output=0.2):
        super().__init__()

        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)
        self.n_emb, self.n_cont = n_emb, n_cont
        self.o_dim = out_features
        # hidden_size = sign_size*cha_input
        hidden_size = 4096
        sign_size1 = sign_size
        sign_size2 = sign_size//2
        output_size = (sign_size//4) * cha_hidden

        self.hidden_size = hidden_size
        self.cha_input = cha_input
        self.cha_hidden = cha_hidden
        self.K = K
        self.sign_size1 = sign_size1
        self.sign_size2 = sign_size2
        self.output_size = output_size
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output

        self.batch_norm1 = nn.BatchNorm1d(self.n_emb + self.n_cont)
        self.dropout1 = nn.Dropout(dropout_input)
        dense1 = nn.Linear(self.n_emb + self.n_cont, hidden_size, bias=False)
        self.dense1 = nn.utils.weight_norm(dense1)

        # 1st conv layer
        self.batch_norm_c1 = nn.BatchNorm1d(cha_input)
        conv1 = conv1 = nn.Conv1d(
            cha_input, 
            cha_input*K, 
            kernel_size=5, 
            stride = 1, 
            padding=2,  
            groups=cha_input, 
            bias=False)
        self.conv1 = nn.utils.weight_norm(conv1, dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = sign_size2)

        # 2nd conv layer
        self.batch_norm_c2 = nn.BatchNorm1d(cha_input*K)
        self.dropout_c2 = nn.Dropout(dropout_hidden)
        conv2 = nn.Conv1d(
            cha_input*K, 
            cha_hidden, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False)
        self.conv2 = nn.utils.weight_norm(conv2, dim=None)

        # 3rd conv layer
        self.batch_norm_c3 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c3 = nn.Dropout(dropout_hidden)
        conv3 = nn.Conv1d(
            cha_hidden, 
            cha_hidden, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False)
        self.conv3 = nn.utils.weight_norm(conv3, dim=None)
        

        # 4th conv layer
        self.batch_norm_c4 = nn.BatchNorm1d(cha_hidden)
        conv4 = nn.Conv1d(
            cha_hidden, 
            cha_hidden, 
            kernel_size=5, 
            stride=1, 
            padding=2, 
            groups=cha_hidden, 
            bias=False)
        self.conv4 = nn.utils.weight_norm(conv4, dim=None)

        self.avg_po_c4 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm2 = nn.BatchNorm1d(output_size)
        self.dropout2 = nn.Dropout(dropout_output)
        dense2 = nn.Linear(output_size, self.o_dim, bias=False)
        self.dense2 = nn.utils.weight_norm(dense2)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x_cont, x_cat):
        x = [e(x_cat[:,i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = torch.cat([x, x_cont], 1)

        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = nn.functional.celu(self.dense1(x))

            

        x = self.batch_norm_c1(x)
        x = nn.functional.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = nn.functional.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = nn.functional.relu(self.conv3(x))

        x = self.batch_norm_c4(x)
        x = self.conv4(x)
        x =  x + x_s
        x = nn.functional.relu(x)

        x = self.avg_po_c4(x)

        x = self.flt(x)

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.dense2(x)

        return x
    

class SoftOrdering1DCNN(pl.LightningModule):

    def __init__(self, in_features, out_features, n_cont, sign_size=32, cha_input=256, cha_hidden=32, 
                 K=2, dropout_input=0.2, dropout_hidden=0.2, dropout_output=0.2):
        super().__init__()


        self.n_cont = n_cont
        self.o_dim = out_features
        # hidden_size = sign_size*cha_input
        hidden_size = 4096
        sign_size1 = sign_size
        sign_size2 = sign_size//2
        output_size = (sign_size//4) * cha_hidden

        self.hidden_size = hidden_size
        self.cha_input = cha_input
        self.cha_hidden = cha_hidden
        self.K = K
        self.sign_size1 = sign_size1
        self.sign_size2 = sign_size2
        self.output_size = output_size
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output

        self.batch_norm1 = nn.BatchNorm1d(self.n_cont)
        self.dropout1 = nn.Dropout(dropout_input)
        dense1 = nn.Linear(self.n_cont, hidden_size, bias=False)
        self.dense1 = nn.utils.weight_norm(dense1)

        # 1st conv layer
        self.batch_norm_c1 = nn.BatchNorm1d(cha_input)
        conv1 = conv1 = nn.Conv1d(
            in_channels=cha_input, 
            out_channels=512, 
            kernel_size=5)
        self.conv1 = nn.utils.weight_norm(conv1, dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = 8)

        # 2nd conv layer
        self.batch_norm_c2 = nn.BatchNorm1d(512)
        self.dropout_c2 = nn.Dropout(dropout_hidden)
        conv2 = nn.Conv1d(
            in_channels=512, 
            out_channels=512, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False)
        self.conv2 = nn.utils.weight_norm(conv2, dim=None)

        # 3rd conv layer
        self.batch_norm_c3 = nn.BatchNorm1d(512)
        self.dropout_c3 = nn.Dropout(dropout_hidden)
        conv3 = nn.Conv1d(
            in_channels=512, 
            out_channels=512, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False)
        self.conv3 = nn.utils.weight_norm(conv3, dim=None)
        

        # 4th conv layer
        self.batch_norm_c4 = nn.BatchNorm1d(512)
        conv4 = nn.Conv1d(
            in_channels=512, 
            out_channels=512, 
            kernel_size=5, 
            stride=1, 
            padding=2, 
            groups=cha_hidden, 
            bias=False)
        self.conv4 = nn.utils.weight_norm(conv4, dim=None)

        self.mx_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flt = nn.Flatten()

        self.batch_norm2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(dropout_output)
        dense2 = nn.Linear(2048, 206, bias=False)
        self.dense2 = nn.utils.weight_norm(dense2)


        self.batch_norm3 = nn.BatchNorm1d(206)
        dense3 = nn.Linear(206, self.o_dim, bias=False)
        self.dense3 = nn.utils.weight_norm(dense3)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):

        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = nn.functional.celu(self.dense1(x))

        x = x.reshape(x.shape[0], self.cha_input, 16)

        x = self.batch_norm_c1(x)
        x = nn.functional.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = nn.functional.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = nn.functional.relu(self.conv3(x))

        x = self.batch_norm_c4(x)
        x = self.conv4(x)
        x =  x * x_s
        
        x = self.mx_pool(x)

        x = self.flt(x)

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = nn.functional.silu(x)
        x = self.dense2(x)
        x = self.batch_norm3(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        return x