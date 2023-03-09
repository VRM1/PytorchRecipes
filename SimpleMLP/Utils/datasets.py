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
import sys
from itertools import tee

# Create Dataset
class CustomFileLoader(Dataset):
    def __init__(self, files, extension, label_clm, \
                 num_features: List[str] = None, cat_features: List[str] = None):
        self.files = files
        self.ext = extension
        self.num_features, self.cat_features = num_features, cat_features
        self.label_clm = label_clm
        
    def __len__(self):
        return len(self.files)
    

    def __getitem__(self, index):

        data_obj = {'numerical_data':None, \
             'categorical_data':None, 'label':None}
        if self.ext == 'parquet':
            data = pd.read_parquet(self.files[index])
        else:
            data = pd.read_csv(self.files[index])
        
        if self.num_features:
            self.x_numerical = torch.FloatTensor(data[self.num_features].values)
        if self.cat_features:
            self.x_categ = torch.LongTensor(data[self.cat_features].values)
        self.y = torch.LongTensor(data[self.label_clm].values).flatten()

        if self.num_features and self.cat_features:
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
class LazyNumericalDataIterator:
    def __init__(self, data, type):
        self.data = data
        self.type = type
    def __iter__(self):
        for d in self.data:
            yield d[self.type]

    def __len__(self):
        return len(self.data)
class LazyLabelDataIterator:
    def __init__(self, data):
        self.data = data
    def __iter__(self):
        for d in self.data:
            yield d['label']

    def __len__(self):
        return len(self.data)
    
class CustomDataLoader(Dataset):
    def __init__(self, data: List):
        
        # unpack data
        # data = list(data)
        data_cat, data_label = tee(data), tee(data)
        self.c_data = None
        self.n_data = ConcatDataset(LazyNumericalDataIterator(data, 'numerical_data'))
        # self.n_data = ConcatDataset([d['numerical_data'] \
        #                               for d in data])
        # if len(data[0]['categorical_data']):
        #     self.c_data = ConcatDataset(LazyNumericalDataIterator(data, 'categorical_data'))
            # self.c_data = ConcatDataset([d['categorical_data'] \
            #                             for d in data])
        # self.label = ConcatDataset([d['label'] \
        #                               for d in data])
        self.label = ConcatDataset(LazyLabelDataIterator(data_label))

    def __getitem__(self, index):

        if self.n_data is not None and self.c_data is not None:
            return self.n_data[index], self.c_data[index], self.label[index]
        else:
            return self.n_data[index], self.label[index]

    def __len__(self):
        return len(self.n_data)

def collate_irregular_batch(batch):
    data = {'numerical_data':None, \
             'categorical_data':None, 'label':None}
    n_data = torch.cat([b['numerical_data'] for b in batch], axis=0)
    c_data = torch.cat([b['categorical_data'] for b in batch], axis=0)
    l_data = torch.cat([b['label'] for b in batch], axis=0)
    data['numerical_data'] = n_data
    data['categorical_data'] = c_data
    data['label'] = l_data
    return data

class MyDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.train_dir = args.train_path
        self.valid_dir = args.valid_path
        self.test_dir = args.test_path
        self.extension = args.extension
        self.num_features: List(str) = None
        self.cat_features: List(str) = None
        self.emb_siz: List(tuple(str, int)) = None
        self.label_clm = args.label_clm
        self.batch_size = args.b_sz
        if args.num_feat_path is not None:
            self.num_features = list(pd.read_csv(args.num_feat_path).columns)
        if args.categ_feat_path is not None:
            self.cat_features = list(pd.read_csv(args.categ_feat_path).columns)
    def _read_train_files(self):

        for data in self.train_file_loader:
            yield data
    
    def _read_valid_files(self):

        for data in self.valid_file_loader:
            yield data
    
    def _read_test_files(self):

        for data in self.test_file_loader:
            yield data
    
    def setup(self, stage=None):

        if stage == "fit":
            t_files = [os.path.join(self.train_dir, f) for f in listdir(self.train_dir) \
                        if f.endswith(self.extension)]
            v_files = [os.path.join(self.valid_dir, f) for f in listdir(self.valid_dir) \
                        if f.endswith(self.extension)]
            self.train_file_loader = DataLoader(CustomFileLoader(t_files, self.extension, self.label_clm, \
                 self.num_features, self.cat_features), batch_size=10, \
                      num_workers=4, collate_fn=collate_irregular_batch)
            self.valid_file_loader = DataLoader(CustomFileLoader(v_files, self.extension, self.label_clm, \
                 self.num_features, self.cat_features), batch_size=10, \
                      num_workers=4, collate_fn=collate_irregular_batch)
            
        if stage == 'test':
            te_files = [os.path.join(self.test_dir, f) for f in listdir(self.test_dir) \
                        if f.endswith(self.extension)]
            self.test_file_loader = DataLoader(CustomFileLoader(te_files, self.extension, self.label_clm, \
                 self.num_features, self.cat_features), batch_size=10, \
                      num_workers=4, collate_fn=collate_irregular_batch)
    def train_dataloader(self):

        data_list = self._read_train_files()
        return DataLoader(CustomDataLoader(data_list), \
                           batch_size=self.batch_size, num_workers=6, prefetch_factor=64)

    
    def val_dataloader(self):

        data_list = self._read_valid_files()
        return DataLoader(CustomDataLoader(data_list), \
                    batch_size=self.batch_size, num_workers=6, prefetch_factor=64)
    
    def test_dataloader(self):

        data_list = self._read_test_files()
        return DataLoader(CustomDataLoader(data_list), \
                    batch_size=self.batch_size, num_workers=6, prefetch_factor=64)
    
    def predict_dataloader(self):

        data_list = self._read_test_files()
        return DataLoader(CustomDataLoader(data_list), \
                    batch_size=self.batch_size, num_workers=6, prefetch_factor=64)
    
class DataRepo:

  def __call__(self, args):

    '''
    Use just one file of data in-order to get the input dimension and 
    embedding size.
    '''
    emb_siz = None
    if  args.categ_feat_path is not None:
        cat_features = list(pd.read_csv(args.categ_feat_path).columns)
        num_categ = pd.read_csv(args.categ_feat_path).values.flatten()
        emb_siz = [(c, min(50, (c+1)//2)) for c in num_categ]
    if args.num_feat_path is not None:
        num_features = list(pd.read_csv(args.num_feat_path).columns)
    
    if args.num_feat_path and args.categ_feat_path:
            i_dim = (len(num_features), len(cat_features))
    elif args.num_feat_path:
        i_dim = len(num_features)
    else:
        i_dim = len(cat_features)
    
    dm = MyDataModule(args)
    return i_dim, emb_siz, dm
