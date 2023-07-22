import torch
import sklearn
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import pytorch_lightning as pl
from os import listdir
import os
import sys
from itertools import tee
from typing import Dict, List
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info
import pdb
from tqdm.auto import tqdm

'''
CustomFileLoader (class): A custom dataset class used to load CSV or Parquet files,
separate numerical and categorical features, and transform the data into PyTorch tensors.

Parameters:
    files:
        A list of file paths containing the data to be loaded.
    extension:
        A string indicating the type of file extension of the data to be loaded. 
        Can be either 'csv' or 'parquet'.
    label_clm:
        A string indicating the column name of the label/target variable in the dataset.
    num_features:
        A list of strings indicating the column names of the numerical features in the dataset. 
        If None, then there are no numerical features in the dataset.
    cat_features:
        A list of strings indicating the column names of the categorical features in the dataset. 
        If None, then there are no categorical features in the dataset.
'''
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

'''
CustomDataLoader (class): A custom dataset class used to combine the numerical and categorical data
from multiple CustomFileLoader objects, and organize them into batches for training or inference.

Parameters:
    data: 
        A list of instances of the CustomFileLoader class, each containing a subset of the data to be concatenated.
'''
class CustomDataLoader(Dataset):
    def __init__(self, data: List):
        
        # data = next(data)
        self.c_data = None
        if len(data['categorical_data']):
            self.c_data = data['categorical_data']
        self.n_data = data['numerical_data']
        self.label = data['label']

    def __getitem__(self, index):

        if self.n_data is not None and self.c_data is not None:
            return self.n_data[index], self.c_data[index], self.label[index]
        else:
            return self.n_data[index], self.label[index]

    def __len__(self):
        return len(self.n_data)

'''
collate_irregular_batch (function): A custom collate function used to handle irregular batch sizes,
by concatenating the numerical and categorical data tensors into a dictionary.
'''
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
'''
GenericDataModule (class):A PyTorch Lightning data module class used to define the training,
    validation, and test datasets, and organize them into batches for
    training or inference. The class also defines properties for the input
    dimension and embedding size of the data.

Constructor Variables
---------------------
    train_dir: 
        A string representing the path to the training dataset 
    directory.

    valid_dir: 
        A string representing the path to the validation dataset 
        directory.

    test_dir: 
        A string representing the path to the test dataset directory.

    extension: 
        A string representing the file extension of the dataset files.

    num_features: 
        A list of strings representing the names of numeric 
        features in the dataset.

    cat_features: 
        A list of strings representing the names of categorical 
        features in the dataset.

    _emb_siz: 
        A list of tuples containing string feature names and their 
        corresponding embedding sizes.

    label_clm: 
        A string representing the name of the label column in the 
        dataset.

    batch_size: 
        An integer representing the batch size for training and 
        testing.

    _input_dim: 
        A list of two integers representing the input dimensions of 
        the dataset.

    _emb_size: 
        A list of tuples containing integer feature indices and their 
        corresponding embedding sizes. The indices correspond to the categorical 
        feature indices after they have been transformed into a continuous 
        numerical representation.

'''
class GenericDataModule():
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_dir = args.train_path
        self.valid_dir = args.valid_path
        self.test_dir = args.test_path
        self.extension = args.extension
        self.num_features: List(str) = None
        self.cat_features: List(str) = None
        self.label_clm = args.label_clm
        self.batch_size = args.b_sz
        self.file_batch_size = args.file_b_sz
        self._input_dim: List(int, int) = (0, 0)
        self._emb_size: List(tuple(int, int)) = (0, 0)
        self.super_buoy = False
        self.valid_data_loader = None
        self.train_data_loader = None
        self.test_data_loader = None
        if args.num_feat_path is not None:
            self.num_features = list(pd.read_csv(args.num_feat_path).columns)
            if self.label_clm in self.num_features:
                self.num_features.remove(self.label_clm)
        if args.categ_feat_path is not None:
            self.cat_features = list(pd.read_csv(args.categ_feat_path).columns)
            if self.label_clm in self.cat_features:
                self.cat_features.remove(self.label_clm)
            
        self.read_files()

    def read_files(self):
        if self.args.inference_mode:
                    self.te_files = [os.path.join(self.test_dir, f) for f in listdir(self.test_dir) \
                        if f.endswith(self.extension)]
        else:
            self.t_files = [os.path.join(self.train_dir, f) for f in listdir(self.train_dir) \
                            if f.endswith(self.extension)]
            self.v_files = [os.path.join(self.valid_dir, f) for f in listdir(self.valid_dir) \
                        if f.endswith(self.extension)]

    
    @property 
    def input_dim(self):
        if self.args.num_feat_path and self.args.categ_feat_path:
            i_dim = (len(self.num_features), len(self.cat_features))
        elif self.args.num_feat_path:
            i_dim = (len(self.num_features), 0)
        else:
            i_dim = (0, len(self.cat_features))
        return i_dim
    
    @property 
    def emb_size(self):
        if self.args.categ_feat_path:
            num_categ = pd.read_csv(self.args.categ_feat_path).values.flatten()
            self._emb_size = [(c, min(50, (c+1)//2)) for c in num_categ]
        return self._emb_size
        


    def setup(self, stage=None):

        if stage == "fit":
            self.train_file_loader = DataLoader(CustomFileLoader(self.t_files, self.extension, self.label_clm, \
                 self.num_features, self.cat_features), batch_size=self.file_batch_size, \
                      num_workers=4, collate_fn=collate_irregular_batch)
            self.valid_file_loader = DataLoader(CustomFileLoader(self.v_files, self.extension, self.label_clm, \
                 self.num_features, self.cat_features), batch_size=self.file_batch_size, \
                      num_workers=4, collate_fn=collate_irregular_batch)
            
        else:
            
            self.test_file_loader = DataLoader(CustomFileLoader(self.te_files, self.extension, self.label_clm, \
                 self.num_features, self.cat_features), batch_size=self.file_batch_size, \
                      num_workers=4, collate_fn=collate_irregular_batch)
    
    def train_dataloader(self):

        return self.train_file_loader

    
    def val_dataloader(self):

        return self.valid_file_loader

    
    def test_dataloader(self):

        
        return DataLoader(CustomDataLoader(self.test_file_loader), \
                    batch_size=self.batch_size, num_workers=6, pin_memory=True, prefetch_factor=64)
    
    def predict_dataloader(self):

        data_list = self._read_test_files()
        return DataLoader(CustomDataLoader(data_list), \
                    batch_size=self.batch_size, num_workers=6, pin_memory=True, prefetch_factor=64)



class WideNDeepDataLoader(GenericDataModule):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._clm_indx: Dict = {}
        self._emb_size: List[tuple(int, int)] = [(0, 0)]
    
    @property
    def clm_indx(self):

        
        if self.args.num_feat_path and self.args.categ_feat_path:
            df = pd.DataFrame(columns=self.num_features+self.cat_features)
        elif self.args.num_feat_path:
            df = pd.DataFrame(columns=self.num_features)
        else:
            df = pd.DataFrame(columns=self.cat_features)
        cols = list(df.columns)
        for c in cols:
            self._clm_indx[c] = df.columns.get_loc(c)
        return self._clm_indx
    
    @property 
    def emb_size(self):
        if self.args.categ_feat_path is not None:
            clms = list(pd.read_csv(self.args.categ_feat_path).columns)
            num_categ = pd.read_csv(self.args.categ_feat_path).values.flatten()
            self._emb_size = [(n, c, min(50, (c+1)//2)) for n, c in zip(clms, num_categ)]
        return self._emb_size
    
    # @property
    # def num_features(self):
    #     return self.num_features

    
class DataRepo:

  def __call__(self, args):

    '''
    Use just one file of data in-order to get the input dimension and 
    embedding size.
    '''
    tab_models = set(['tabmlp','tabresnet', 'fttransformer'])
    if args.model in tab_models:
        dm = WideNDeepDataLoader(args)
    else:
        dm = GenericDataModule(args)
    
    return dm
