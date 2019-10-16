import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self,i_dim, o_dim, h_dim,d_rate):

        super().__init__()
        # input dimension
        self.i_dim = i_dim
        # output dimension
        self.o_dim = o_dim
        # number of hidden dimensions
        self.h_dim = h_dim
        # dropout rate
        self.d_rate = d_rate
        # define a fully connected layer
        self.l1 = nn.Linear(i_dim,h_dim)
        # define the dropout layer
        self.l1_drop = nn.Dropout(d_rate)
        # define a second FCN
        self.l2 = nn.Linear(h_dim,h_dim)
        # define the second dropout layer
        self.l2_drop = nn.Dropout(d_rate)
        # define the final prediction layer
        self.ol = nn.Linear(h_dim,o_dim)

    def forward(self,x):

        x = F.relu(self.l1(x))
        x = self.l1_drop(x)
        x = F.relu(self.l2(x))
        x = self.l2_drop(x)
        # not sure why we give dim=1
        return F.log_softmax(self.ol(x),dim=1)
