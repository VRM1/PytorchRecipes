import torch
import sklearn
from sklearn.datasets import load_breast_cancer
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import pandas as pd
# needs sklearn 0.23 +
DATA_LUKUP = {'breast_cancer':load_breast_cancer(as_frame=True)}

class old_breastCancerDataset(torch.utils.data.Dataset):
  def __init__(self):
    data = load_breast_cancer()

    self.X = data.data
    self.y = data.target
    self.shape = self.X.shape

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

  def __len__(self):
    return len(self.X)

class custom_data_loader(torch.utils.data.Dataset):

  def __init__(self, df):
    # self.X = torch.FloatTensor(df.loc[:, df.columns != 'label'].values)
    self.X = df.loc[:, df.columns != 'label']
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

  def __call__(self, name, is_valid=False, train_batch_sz=256, test_batch_sz=512):

    data = DATA_LUKUP[name]
    if isinstance(data, sklearn.utils.Bunch):
      df = pd.DataFrame(data.data, columns=data.feature_names)
      df['label'] = pd.Series(data.target)

    i_channel = 1
    n_classes = len(df['label'].unique())
    # if n_classes <= 2:
    #   n_classes = 1
    train_d = custom_data_loader(df)
    i_dim = train_d.shape[1]
    train_len = len(train_d)
    split_t = int(np.floor(0.4 * train_len))
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

  name = 'breast_cancer'
  GetData(name)
