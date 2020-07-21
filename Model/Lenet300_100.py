import torch
import torch.nn as nn
import torch.nn.functional as F

class Lenet300_100(nn.Module):

    def __init__(self,in_features, out_features,d_rate=0.2):

        super().__init__()
        # input dimension
        self.i_dim = in_features
        # output dimension
        self.o_dim = out_features
        # dropout rate
        self.d_rate = d_rate
        # define a fully connected layer
        self.l1 = nn.Linear(self.i_dim ,512)
        # define the dropout layer
        # define a second FCN
        self.l2 = nn.Linear(512,256)
        # define the final prediction layer
        self.ol = nn.Linear(256,self.o_dim)

    def forward(self,x):

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.ol(x)

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
