import torch
import sklearn
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
from tqdm import tqdm
from typing import List
import pytorch_lightning as pl
from os import listdir
import os


# Create Dataset
class CustomFileLoader(Dataset):
    def __init__(self, files, extension, label_clm, \
                 num_feat_path=str, categ_feat_path=str):
        self.files = files
        self.ext = extension
        self.num_feat_path, self.categ_feat_path = \
             num_feat_path, categ_feat_path
        self.num_features, self.categ_features = List[str], List[str]
        self.label_clm = label_clm
        self.label_indx, self.__emb_siz = None, None
        
    def __len__(self):
        return len(self.files)
    
    def process(self, data):

        if self.num_feat_path:
          req_features = pd.read_csv(self.num_feat_path).columns
          self.num_features = self.__get_clm_indices(data, req_features)
        if self.categ_feat_path:
            req_features = pd.read_csv(self.categ_feat_path).columns
            self.num_categ = pd.read_csv(self.categ_feat_path).values.flatten()
            self.cat_features = self.__get_clm_indices(data, req_features)
            self.__emb_siz = [(c, min(50, (c+1)//2)) for c in self.num_categ]
        
        self.label_clm = self.__get_clm_indices(data, [self.label_clm])
    
    def __get_clm_indices(self, data, feat_list):
        header = [data.columns.get_loc(c) for c in feat_list]
        return header

    def __getitem__(self, index):

        data_obj = {'numerical_data':None, \
             'categorical_data':None, 'label':None}
        if self.ext == 'parquet':
            data = pd.read_parquet(self.files[index])
        else:
            data = pd.read_csv(self.files[index])
        self.process(data)
        if self.num_feat_path:
            self.x_numerical = torch.FloatTensor(data.iloc[:,self.num_features].values)
        if self.categ_feat_path:
            self.x_categ = torch.LongTensor(data.iloc[:,self.cat_features].values)
        self.y = torch.LongTensor(data.iloc[:,self.label_clm].values).flatten()

        if self.num_feat_path and self.categ_feat_path:
            data_obj['numerical_data'] = self.x_numerical
            data_obj['categorical_data'] = self.x_categ
            data_obj['label'] = self.y
            return data_obj
        else:
            data_obj['numerical_data'] = self.x_numerical
            # fill up with random value for categorical to avoid pytorch returning error due to None value
            data_obj['categorical_data'] = torch.empty(0)
            data_obj['label'] = self.y
            return data_obj
    
    @property
    def emb_size(self):
        return self.__emb_siz

    @property
    def feature_size(self):
        if self.num_feat_path and self.categ_feat_path:
            return (len(self.num_features), len(self.cat_features))
        elif self.num_feat_path:
            return len(self.num_features)
        else:
            return len(self.cat_features)

class CustomDataLoader(Dataset):
    def __init__(self, data: dict):

        self.n_data, self.c_data = None, None
        l_data = data['label']
        self.l_data = l_data.view(l_data.shape[0]*l_data.shape[1])
        if data['numerical_data'].shape[1] and data['categorical_data'].shape[1]:
            n_data = data['numerical_data']
            c_data = data['categorical_data']
            self.n_data = n_data.view(n_data.shape[0]*n_data.shape[1], -1)
            self.c_data = c_data.view(c_data.shape[0]*c_data.shape[1], -1)

        else:
            n_data = data['numerical_data']
            self.n_data = n_data.view(n_data.shape[0]*n_data.shape[1], -1)

    def __getitem__(self, index):

        if self.n_data is not None and self.c_data is not None:
            return self.n_data[index], self.c_data[index], self.l_data[index]
        else:
            return self.n_data[index], self.l_data[index]

    def __len__(self):
        return len(self.n_data)


class MyDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.train_dir = args.train_path
        self.valid_dir = args.valid_path
        self.test_dir = args.test_path
        self.extension = args.extension
        self.num_feat_path = args.num_features
        self.cat_feat_path = args.cat_features
        self.label_clm = args.label_clm
        self.batch_size = args.b_sz
    
    def setup(self, stage=None):

        if stage == "fit":
            t_files = [os.path.join(self.train_dir, f) for f in listdir(self.train_dir) \
                        if f.endswith(self.extension)]
            v_files = [os.path.join(self.valid_dir, f) for f in listdir(self.valid_dir) \
                        if f.endswith(self.extension)]
            tfl = ConcatDataset([CustomFileLoader(t_files, self.extension, self.label_clm, \
                 self.num_feat_path, self.cat_feat_path) for _ in range(len(t_files))])
            
            vfl = ConcatDataset([CustomFileLoader(v_files, self.extension, self.label_clm, \
                 self.num_feat_path, self.cat_feat_path)for _ in range(len(v_files))])
            
            self.train_file_loader = DataLoader(tfl, num_workers=4)
            self.valid_file_loader = DataLoader(vfl, num_workers=4)
        if stage == 'test':
            te_files = [os.path.join(self.test_dir, f) for f in listdir(self.test_dir) \
                        if f.endswith(self.extension)]
            flte = CustomFileLoader(te_files, self.extension, self.label_clm, \
                 self.num_feat_path, self.cat_feat_path) 
            self.test_file_loader = DataLoader(flte, batch_size=10, num_workers=4)
    
    def train_dataloader(self):

        for f in self.train_file_loader:
            data_loader = DataLoader(CustomDataLoader(f), batch_size=self.batch_size, num_workers=6)
            return data_loader
    
    def val_dataloader(self):

        for f in self.valid_file_loader:
            data_loader = DataLoader(CustomDataLoader(f), batch_size=self.batch_size, num_workers=6)
            return data_loader
    
    def test_dataloader(self):

        for f in self.test_file_loader:
            data_loader = DataLoader(CustomDataLoader(f), batch_size=self.batch_size, num_workers=6)
            return data_loader
    
    def predict_dataloader(self):

        for f in self.test_file_loader:
            data_loader = DataLoader(CustomDataLoader(f), batch_size=self.batch_size, num_workers=6)
            return data_loader

    

class DataRepo:

  def __call__(self, args):

    '''
    Use just one file of data in-order to get the input dimension and 
    embedding size.
    '''
    file = [os.path.join(args.train_path, f) \
             for f in listdir(args.train_path) \
                  if f.endswith(args.extension)][:1]
    fl = CustomFileLoader(file, args.extension, args.label_clm, \
                 num_feat_path=args.num_features, categ_feat_path=args.cat_features)
    obj_data = DataLoader(fl)
    for f in obj_data:
        emb_size = fl.emb_size
        i_dim = fl.feature_size
    data = MyDataModule(args)
    return  i_dim, emb_size, data

