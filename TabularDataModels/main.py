import torch
import os
from argparse import ArgumentParser
from Utils import initialize_arguments, DataRepo
import torch.nn as nn
from Model import Mlp, BayesianMlp, TabMLP, TResnet
from Model import FTransformer, Conv, AttentionMLP
from Model import SoftOrdCNN
from Utils import EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score, roc_auc_score
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pdb
torch.set_float32_matmul_precision('medium')
warnings.filterwarnings('ignore', category=UserWarning)

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.manual_seed(112)

if torch.cuda.is_available():
    DEVICE = 'gpu'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

if not os.path.isdir('trained_weights'):
    os.makedirs('trained_weights')


class RunModel:

    def __init__(self, args):

        self.args = args
        self.tr_b_sz = args.b_sz
        self.tst_b_sz = 512
        self.test_mode = args.test
        self.is_bayesian = args.is_bayesian
        # number of MCMC samples for Bayesian NN. If network is not bayesian, it is simply set as 1
        if args.is_bayesian:
            self.n_samples = 3
        else:
            self.n_samples = 1
        self.optim = args.optimizer
        self.m_name = args.model
        self.lr = args.learning_rate
        self.resume = args.resume
        self.start_epoch = 0
        self.n_classes = args.n_classes
        self.batch_size = args.b_sz
        self.workers = args.data_workers
        self.epoch = args.epochs
        data_repo = DataRepo()
        self.dl = data_repo(args)

        if self.n_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.init_model()

    def init_model(self):

        if self.m_name == 'mlp':
            self.model = Mlp(epochs=self.epoch, in_features=self.dl.input_dim, out_features=self.n_classes,
                              batch_size=self.batch_size, workers=self.workers,
                               learning_rate=self.lr, emb_size=self.dl.emb_size,
                                 cat_features=self.args.categ_feat_path, args=self.args)
        elif self.m_name == 'bmlp':
            self.model = BayesianMlp(epochs=self.epoch, in_features=self.dl.input_dim, out_features=self.n_classes,
                              batch_size=self.batch_size, workers=self.workers,
                               learning_rate=self.lr, emb_size=self.dl.emb_size,
                                 cat_features=self.args.categ_feat_path, args=self.args)
        elif self.m_name == 'cnn':
            self.model = Conv(self.dl.input_dim, self.n_classes,
                              self.dl.emb_size, self.args.categ_feat_path)
        elif self.m_name == 'soft_cnn':
            self.model = SoftOrdCNN(self.dl.input_dim, self.n_classes,
                                    self.dl.emb_size, self.args.categ_feat_path)
        elif self.m_name == 'tabmlp':
            self.model = TabMLP(epochs=self.epoch, in_features=self.dl.input_dim,
                                 out_features=self.n_classes, file_loader=self.dl,
                                  batch_size=self.batch_size, workers=self.workers,
                                  column_indx=self.dl.clm_indx, emb_size=self.dl.emb_size,
                                    num_columns=self.dl.num_features,
                                      cat_features=self.args.categ_feat_path)
        elif self.m_name == 'fttransformer':
            self.model = FTransformer(epochs=self.epoch, in_features=self.dl.input_dim,
                                 out_features=self.n_classes, file_loader=self.dl,
                                  batch_size=self.batch_size, workers=self.workers,
                                  column_indx=self.dl.clm_indx, emb_size=self.dl.emb_size,
                                    num_columns=self.dl.num_features,
                                      cat_features=self.args.categ_feat_path)
        elif self.m_name == 'tabresnet':
            self.model = TResnet(epochs=self.epoch, in_features=self.dl.input_dim,
                                 out_features=self.n_classes, file_loader=self.dl,
                                  batch_size=self.batch_size, workers=self.workers,
                                  column_indx=self.dl.clm_indx, emb_size=self.dl.emb_size,
                                    num_columns=self.dl.num_features,
                                      cat_features=self.args.categ_feat_path)
        elif self.m_name == 'attentionmlp':
            self.model = AttentionMLP(epochs=self.epoch, in_features=self.dl.input_dim,
                                 out_features=self.n_classes, file_loader=self.dl,
                                  batch_size=self.batch_size, workers=self.workers,
                                  column_indx=self.dl.clm_indx, emb_size=self.dl.emb_size,
                                    num_columns=self.dl.num_features,
                                      cat_features=self.args.categ_feat_path)
        print(self.model)
        if self.args.inference_mode:
            self.model = type(self.model).load_from_checkpoint('{}/{}/best.ckpt'. \
                                                               format(self.args.model_storage_path, self.args.model))
            # re-initialize the file loader
            # self.model.file_loader = self.dl
        early_stop_callback = EarlyStopping(monitor="val_loss",patience=self.args.patience, \
                                             verbose=True, mode="min")
        checkpoint_callback = pl_callbacks.ModelCheckpoint(dirpath='{}/{}'. \
                                                           format(self.args.model_storage_path, self.args.model), \
                                                           filename='best', monitor='val_loss', save_last=True)
        if DEVICE == 'gpu':

            self.trainer = pl.Trainer(accelerator=DEVICE, max_epochs=args.epochs, \
                                      min_epochs=1, num_sanity_val_steps=0,
                                      callbacks=[early_stop_callback, checkpoint_callback], )
            current_device = torch.cuda.current_device()
            print(f"PyTorch Lightning is using GPU device {current_device}")
        else:
            self.trainer = pl.Trainer(accelerator=DEVICE, max_epochs=args.epochs, \
                 min_epochs=1, callbacks=[early_stop_callback, checkpoint_callback])

    def train(self):
        if self.args.data_in_memory:

            if args.ckpt_path != 'None':

                # need to modify this line to just selt.train.fit(model)
                self.trainer.fit(model=self.model, ckpt_path=args.ckpt_path, train_dataloaders=self.dl,
                                val_dataloaders=self.dl)
            else:
                self.trainer.fit(model=self.model, train_dataloaders=self.dl)
        else:

            if args.ckpt_path != 'None':

                # need to modify this line to just selt.train.fit(model)
                self.trainer.fit(model=self.model, ckpt_path=args.ckpt_path)
            else:
                self.trainer.fit(model=self.model)

    def get_feature_importance(self, model):
        # Extract mask values for embeddings and continuous features
        # emb_importance = model.model.mask_emb.detach().cpu().numpy()
        # cont_importance = model.model.mask_cont.detach().cpu().numpy()

        emb_importance = F.softmax(model.model.mask_emb, dim=-1).detach().cpu().numpy()
        cont_importance = F.softmax(model.model.mask_cont, dim=-1).detach().cpu().numpy()
        
        # Combine them into a single array
        combined_importance = np.concatenate([emb_importance, cont_importance])
        
        return combined_importance


    def plot_feature_importance(self, importance, feature_names):
        # Sort features based on importance
        sorted_idx = np.argsort(importance)
        
        # Plot
        plt.figure(figsize=(10, 15))
        plt.title("Feature Importance")
        plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel("Importance Value")
        plt.show()

    def test(self, load_best_model=False):

        # self.trainer.test(self.model, self.dl)
        if self.args.data_in_memory:
            preds = self.trainer.predict(self.model, self.dl)
        else:
            preds = self.trainer.predict(self.model)
        y = torch.concat([p[1] for p in preds]).numpy()
        preds = torch.concat([p[0] for p in preds])
        preds = torch.nn.functional.softmax(preds).numpy()
        print(classification_report(y, preds.argmax(axis=1)))
        print("ROC-AUC:{}".format(roc_auc_score(y, preds[:, 1])))
        print("PrecisionRecall-AUC:{}".format(average_precision_score(y, preds[:, 1])))
        feature_names = pd.read_csv(self.args.num_feat_path)

        # importance_values = self.get_feature_importance(self.model)
        # self.plot_feature_importance(importance_values, feature_names)



if __name__ == '__main__':
    seed_everything(42)
    parser = ArgumentParser(description='')
    args = initialize_arguments(parser)
    run_model = RunModel(args)
    start_time = time.time()
    if not args.inference_mode:
        run_model.train()
        # run_model.test()
    else:
        run_model.test()
    print("--- %s seconds ---" % (time.time() - start_time))
