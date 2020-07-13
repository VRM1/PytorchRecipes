import torch
import numpy as np
import pandas as pd
from torchtext.data import TabularDataset
from torchtext.data import Field, BucketIterator
from sklearn.model_selection import train_test_split
import spacy
import gensim
from .EarlyStopping import EarlyStopping
def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    # load the spacy language model, NOTE make sure to download the model in terminal using: python -m spacy download <language name>
    spacy_en = spacy.load('en')
    # tokenize text
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_fr(text):
    """
    Tokenizes french text from a string into a list of strings (tokens)
    """
    # load the spacy language model
    spacy_fr = spacy.load('fr')
    # tokenize text
    return [tok.text for tok in spacy_fr.tokenizer(text)]

def SimpleTok(text):
    return text.split()

def PrepTranslationData(pth):

    batch_sz = 128
    # create a field for source language
    src_txt = Field(sequential=True, tokenize=SimpleTok, init_token = '<sos>', eos_token = '<eos>', lower=True)
    dst_txt = Field(sequential=True, tokenize=SimpleTok, init_token = '<sos>', eos_token = '<eos>', lower=True)
    d_format = [('src_l',src_txt),('tgt_l',dst_txt)]

    trn,val,tst = TabularDataset.splits(path=pth,train='train.csv',validation='valid.csv',\
                                        test='test.csv',format='csv',skip_header=True,fields=d_format)
    # convert words to integers
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('building vocabulary')
    # we pass in the same train data, but how come we are getting two different outputs (need to investigate this)
    src_txt.build_vocab(trn,min_freq=2)
    dst_txt.build_vocab(trn,min_freq=2)
    train_iter = BucketIterator.splits((trn,), batch_size=batch_sz, device=dev)[0]
    val_iter = BucketIterator.splits((val,), batch_size=batch_sz, device=dev)[0]
    tst_iter = BucketIterator.splits((tst,), batch_size=batch_sz, device=dev)[0]
    # train_iter, val_iter, tst_iter = BucketIterator.splits((trn, val, tst), batch_size=batch_sz,\
    #                                 device=dev,repeat=False)
    return(src_txt,dst_txt,train_iter,val_iter,tst_iter)

# method to split the main data into train and test
def SplitTrTest(pth,name):
    data = pd.read_csv(pth+name,sep='\t')
    # enable for small data
    # data = data.iloc[:10000,:]
    train,test = train_test_split(data,train_size=0.8)
    train,valid = train_test_split(data,train_size=0.8)
    train.to_csv(pth+'train.csv',index=False)
    valid.to_csv(pth+'valid.csv',index=False)
    test.to_csv(pth+'test.csv',index=False)

class AmazonReviewDataset(torch.utils.data.Dataset):
  def __init__(self,pth,fil_a,fil_b):
    # import and initialize dataset
    d2v_embeds = gensim.models.doc2vec.Doc2Vec.load(pth+fil_a)
    feats = []
    for i in range(len(d2v_embeds.docvecs)):
        feats.append(d2v_embeds.docvecs[i])

    self.X = torch.FloatTensor(feats)
    self.Y = torch.from_numpy(pd.read_json(pth+fil_b,lines=True)['overall'].values.astype(np.longlong))
    print('test')
  def __getitem__(self, idx):
    # get item by index
    return self.X[idx], self.Y[idx]

  def __len__(self):
    # returns length of data
    return len(self.X)
