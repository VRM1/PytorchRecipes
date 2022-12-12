import torch
import sklearn
from sklearn.datasets import load_breast_cancer
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pathlib
import glob



# needs sklearn 0.23 +
DATA_LUKUP = {'covid19':'Covid19Classification/LitCovid_doc2vec_embeddings.json',
"long_document":"LongDocumentClassification/LongDocumentDataset_doc2vec_embeddings.json",
}
 # Create Dataset
class CustomDataLoader(Dataset):
    def __init__(self, path, nb_samples, label_clm, req_features=None):
        self.df = pd.read_csv(path)
        self.req_features = None
        if req_features:
          req_features = pd.read_csv(req_features).columns
          self.req_features = self.__get_clm_indices(req_features)

        self.label_clm = self.__get_clm_indices([label_clm])
        self.x_numerical = torch.FloatTensor(self.df.iloc[:,self.req_features].values)
        self.y = torch.LongTensor(self.df.iloc[:,self.label_clm].values).flatten()
    def __get_clm_indices(self, feat_list):
        header = [self.df.columns.get_loc(c) for c in feat_list]
        return header
    def __getitem__(self, index):
        
        return self.x_numerical[index], self.y[index]
    def __len__(self):
        return len(self.x_numerical)
    
    @property
    def num_features(self):
      return len(self.req_features)

 # Create Dataset
class CustomLazyDataLoader(Dataset):
    def __init__(self, path, nb_samples, label_clm, req_features=None):
        self.path = path
        self.chunksize = 100
        self.len = nb_samples // self.chunksize
        self.req_features = None
        if req_features:
          req_features = pd.read_csv(req_features).columns
          self.req_features = self.__get_clm_indices(req_features)

        self.label_clm = self.__get_clm_indices([label_clm])

    def __get_clm_indices(self, feat_list):
        header = next(pd.read_csv(self.path, chunksize=1))
        header = [header.columns.get_loc(c) for c in feat_list]
        return header
    def __getitem__(self, index):
        x = next(
            pd.read_csv(
                self.path,
                skiprows=index * self.chunksize + 1,  #+1, since we skip the header
                chunksize=self.chunksize))
        x_numerical = torch.FloatTensor(x.iloc[:,self.req_features].values)
        y = torch.LongTensor(x.iloc[:,self.label_clm].values)

        return x_numerical, y
    def __len__(self):
        return self.len
    
    @property
    def num_features(self):
      return len(self.req_features)

class DataRepo:

  def __call__(self, args, is_valid=False, train_batch_sz=256, test_batch_sz=512):

    # i_channel = 1
    # n_classes = len(df['label'].unique())

    train_d = CustomDataLoader(args.train_path, args.train_size, args.label_clm, args.req_features)
    valid_d = CustomDataLoader(args.valid_path, args.valid_size, args.label_clm, args.req_features)
    test_d = CustomDataLoader(args.test_path, args.test_size, args.label_clm, args.req_features)
    train_loader = DataLoader(train_d, batch_size=train_batch_sz, num_workers=1)
    valid_loader = DataLoader(valid_d, batch_size=test_batch_sz, num_workers=1)
    test_loader = DataLoader(test_d, batch_size=test_batch_sz, num_workers=1)
    i_dim = train_d.num_features
    train_len = args.train_size
    valid_len = args.valid_size
    test_len = args.test_size
    i_channel = 1
    n_classes = args.n_classes
    
    return n_classes, i_channel, i_dim, train_len, valid_len, \
           test_len, train_loader, valid_loader, test_loader

if __name__ == '__main__':

  obj = DataRepo()
  obj('long_document')
  
