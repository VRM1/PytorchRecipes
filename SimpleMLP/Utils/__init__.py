import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
import gensim
from .EarlyStopping import EarlyStopping
from .SummaryWriter import LogSummary
from .ArgumentParser import initialize_arguments

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
