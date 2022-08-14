import torch
import sklearn
from sklearn.datasets import load_breast_cancer
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import pandas as pd
import pdb
import os

DATA_LOC = '/home/vineeth/Documents/DataRepo/'

# needs sklearn 0.23 +
DATA_LUKUP = {'covid19':'Covid19Classification/LitCovid_doc2vec_embeddings.json',
"long_document":"LongDocumentClassification/LongDocumentDataset_doc2vec_embeddings.json"}

class custom_data_loader(torch.utils.data.Dataset):

  def __init__(self, df):
    self.X = df.loc[:, df.columns != 'label']
    # if an unormalized dataset you need to normalize as follows
    self.X = (self.X-self.X.mean())/self.X.std()
    self.X = torch.FloatTensor(self.X.values)
    self.y = torch.LongTensor(df.label.values)
    # self.y = torch.FloatTensor(df.label.values)
    self.shape = self.X.shape
  
  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]
  
  def __len__(self):
    return len(self.X)


class DataRepo:

  def __call__(self, args, name, is_valid=False, train_batch_sz=256, test_batch_sz=512):

    global DATA_LOC
    DATA_LOC = args.data_path
    if name == 'breast_cancer':
      data = load_breast_cancer()
      df = pd.DataFrame(data.data, columns=data.feature_names)
      df['label'] = pd.Series(data.target)
    else:
      data = pd.read_json(DATA_LOC+DATA_LUKUP[name], lines=True)
      df = pd.DataFrame(data["embedings"].to_list(), columns=['feature'+str(i) for i in range(len(data["embedings"].iloc[0]))])
      df['label'] = pd.Series(data.label)

    i_channel = 1
    n_classes = len(df['label'].unique())
    # if n_classes <= 2:
    #   n_classes = 1
    train_d = custom_data_loader(df)
    i_dim = train_d.shape[1]
    train_len = len(train_d)
    split_t = int(np.floor(0.2 * train_len))
    indices = range(train_len)
    if is_valid:
        split_v = int(np.floor(0.1 * train_len))
    else:
        split_v = int(np.floor(0 * train_len))

    test_indx = np.random.choice(indices, split_t, replace=False)
    indices = list(set(indices).difference(set(test_indx)))
    valid_indx = np.random.choice(indices, split_v, replace=False)

    indices = list(set(indices).difference(set(valid_indx)))
    train_sampler = SubsetRandomSampler(indices)
    valid_sampler = SubsetRandomSampler(valid_indx)
    test_sampler = SubsetRandomSampler(test_indx)
    train_len = len(indices)
    valid_len = len(valid_indx)
    test_len = len(test_indx)
    assert train_len + valid_len + test_len == len(train_d)
    train_loader = DataLoader(train_d, batch_size=train_batch_sz, sampler=train_sampler, num_workers=4)
    valid_loader = DataLoader(train_d, batch_size=test_batch_sz, sampler=valid_sampler, num_workers=4)
    test_loader = DataLoader(train_d, batch_size=test_batch_sz, sampler=test_sampler, num_workers=4)

    return n_classes, i_channel, i_dim, train_len, valid_len, \
           test_len, train_loader, valid_loader, test_loader

if __name__ == '__main__':

  obj = DataRepo()
  obj('long_document')
  
