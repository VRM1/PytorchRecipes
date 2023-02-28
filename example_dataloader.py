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

def collate_fn(batch):
    # Get the length of each tensor in the batch
    lengths = [len(sample['numerical_data']) for sample in batch]

    # Get the size of the longest tensor in the batch
    max_length = max(lengths)

    # Create empty tensors to hold the padded data
    numerical_data_padded = torch.zeros((len(batch), max_length, batch[0]['numerical_data'].shape[1]))
    categorical_data_padded = torch.zeros((len(batch), max_length, batch[0]['categorical_data'].shape[1]))

    # Copy the data from each sample into the padded tensors
    for i, sample in enumerate(batch):
        numerical_data_padded[i, :lengths[i], :] = sample['numerical_data']
        categorical_data_padded[i, :lengths[i], :] = sample['categorical_data']

    # Pad and stack the label tensors
    label_padded = []
    for i, sample in enumerate(batch):
        label = sample['label']
        padding_length = max_length - len(label)
        label_padded.append(torch.cat([label, torch.zeros(padding_length, dtype=torch.long) + i]))
    label = torch.stack(label_padded)

    # Return a dictionary containing the padded tensors and the label tensor
    return {'numerical_data': numerical_data_padded, 'categorical_data': categorical_data_padded, 'label': label}


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
            data_obj['categorical_data'] = torch.zeros((1, 1))
            data_obj['label'] = self.y
            return data_obj
    
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

if __name__ == '__main__':
    parquet_dir = '/media/6TB_Volume/DataRepo/LargeSyntheticData/tmp'
    parquet_files = ['{}/{}'.format(parquet_dir, f) for f in listdir(parquet_dir) if f.endswith('parquet')]
    num_feat_path = '/media/6TB_Volume/DataRepo/LargeSyntheticData/numerical_clms.csv'
    categ_feat_path = None
    ext = 'parquet'
    if num_feat_path is not None:
        num_features = list(pd.read_csv(num_feat_path).columns)
    if categ_feat_path is not None:
        cat_features = list(pd.read_csv(categ_feat_path).columns)
        num_categ = pd.read_csv(categ_feat_path).values.flatten()
        emb_siz = [(c, min(50, (c+1)//2)) for c in num_categ]

    concat_dataset = ConcatDataset([CustomFileLoader(parquet_files, ext, label_clm='label', \
                                                      num_features=num_features, cat_features=None)])
    
    # Define DataLoader
    batch_size = 32
    shuffle = True
    num_workers = 4
    data_loader = DataLoader(concat_dataset, batch_size=5, \
                            shuffle=shuffle, num_workers=1, collate_fn=collate_fn)
    with tqdm(total=737867) as pbar:
        cnt = 0
        for batch_indx, batch_data in enumerate(data_loader):
            data_loader = DataLoader(CustomDataLoader(batch_data), batch_size=256, num_workers=5)
            for row in data_loader:
                cnt += row[0].shape[0]
                pbar.update(row[0].shape[0])