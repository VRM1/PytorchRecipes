import torch
import sklearn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# needs sklearn 0.23 +
DATA_LUKUP = {'covid19':'Covid19Classification/LitCovid_doc2vec_embeddings.json',
"long_document":"LongDocumentClassification/LongDocumentDataset_doc2vec_embeddings.json",
}
 # Create Dataset
class CustomDataLoader(Dataset):
    def __init__(self, path, label_clm, num_features=None, cat_features=None):
        self.df = pd.read_csv(path)
        self.num_features, self.cat_features = None, None
        self.x_numerical, self.x_categ = None, None
        if num_features:
          req_features = pd.read_csv(num_features).columns
          self.num_features = self.__get_clm_indices(req_features)
          self.x_numerical = torch.FloatTensor(self.df.iloc[:,self.num_features].values)
        if cat_features:
            req_features = pd.read_csv(cat_features).columns
            self.cat_features = self.__get_clm_indices(req_features)
            self.x_categ = torch.LongTensor(self.df.iloc[:,self.cat_features].values)

        self.label_clm = self.__get_clm_indices([label_clm])
        self.y = torch.LongTensor(self.df.iloc[:,self.label_clm].values).flatten()
    def __get_clm_indices(self, feat_list):
        header = [self.df.columns.get_loc(c) for c in feat_list]
        return header
    def __getitem__(self, index):
        
        if self.cat_features:
            return self.x_numerical[index], self.x_categ[index], self.y[index]
        else:
            return self.x_numerical[index], self.y[index]
    @property
    def feature_size(self):
        if self.num_features and self.cat_features:
            return len(self.num_features) + len(self.cat_features)
        elif self.num_features:
            return len(self.num_features)
        else:
            return len(self.cat_features)

    def __len__(self):
        return len(self.df)
    

class DataRepo:

  def __call__(self, args, is_valid=False, train_batch_sz=256, test_batch_sz=512):

    # i_channel = 1
    # n_classes = len(df['label'].unique())

    train_d = CustomDataLoader(args.train_path, \
         args.label_clm, args.num_features, args.cat_features)
    valid_d = CustomDataLoader(args.valid_path, \
         args.label_clm, args.num_features, args.cat_features)
    test_d = CustomDataLoader(args.test_path, \
         args.label_clm, args.num_features, args.cat_features)
    train_loader = DataLoader(train_d, batch_size=train_batch_sz, num_workers=1)
    valid_loader = DataLoader(valid_d, batch_size=test_batch_sz, num_workers=1)
    test_loader = DataLoader(test_d, batch_size=test_batch_sz, num_workers=1)
    i_dim = train_d.feature_size
    train_len = len(train_d)
    valid_len = len(valid_d)
    test_len = len(test_d)
    i_channel = 1
    n_classes = args.n_classes
    
    return n_classes, i_channel, i_dim, train_len, valid_len, \
           test_len, train_loader, valid_loader, test_loader

if __name__ == '__main__':

  obj = DataRepo()
  obj('long_document')
  
