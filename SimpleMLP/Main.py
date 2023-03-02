from argparse import ArgumentParser
import torch
from Utils import initialize_arguments, DataRepo
import torch.nn as nn
from Model import SimpleLenet
import os
from Utils import EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score, roc_auc_score
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import time
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
        self.epochs = 150
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
        data_repo = DataRepo()
        self.i_dim, self.emb_size, \
            self.data_loader = data_repo(args)
        
        if self.n_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.init_model()


    def init_model(self):

        if self.m_name == 'lenet300-100':
            self.model = SimpleLenet(self.i_dim, self.n_classes, \
                 self.emb_size, self.args.cat_features)
        if self.args.inference_mode:
            self.model = type(self.model).load_from_checkpoint('{}/best.ckpt'. \
                format(self.args.model_storage_path))
        early_stop_callback = EarlyStopping(monitor="val_loss", \
             min_delta=0.01, patience=self.args.patience, verbose=False, mode="min")
        checkpoint_callback = pl_callbacks.ModelCheckpoint(dirpath=self.args.model_storage_path, filename='best', \
            monitor='val_loss', save_last=True)
        if DEVICE == 'gpu':
            self.trainer = pl.Trainer(accelerator=DEVICE, max_epochs=args.epochs, \
                 min_epochs=1, callbacks=[early_stop_callback, checkpoint_callback])
        else:
            self.trainer = pl.Trainer(accelerator=DEVICE, max_epochs=args.epochs, \
                 min_epochs=1, callbacks=[early_stop_callback, checkpoint_callback])

    
    def train(self):
        
        if args.ckpt_path != 'None':
            # self. trainer.fit(max_epochs=100, min_epochs=1, model=self.model, train_dataloaders=self.data, \
            #         val_dataloaders=self.valid_loader, ckpt_path=args.ckpt_path)
            self. trainer.fit(max_epochs=100, min_epochs=1, model=self.model, train_dataloaders=self.data_loader, \
                    val_dataloaders=self.data_loader, ckpt_path=args.ckpt_path)
        else:
            self.trainer.fit(model=self.model, train_dataloaders=self.data_loader)
    
    def test(self, load_best_model=False):

        self.trainer.test(self.model, self.data_loader)
        preds  = self.trainer.predict(self.model, self.data_loader)
        y = torch.concat([p[1] for p in preds]).numpy()
        preds = torch.concat([p[0] for p in preds])
        preds = torch.nn.functional.softmax(preds).numpy()
        print(classification_report(y, preds.argmax(axis=1)))
        print("ROC-AUC:{}".format(roc_auc_score(y, preds[:, 1])))
        print("PrecisionRecall-AUC:{}".format(average_precision_score(y, preds[:, 1])))



if __name__ == '__main__':
    seed_everything(42)
    parser = ArgumentParser(description='')
    args = initialize_arguments(parser)
    run_model = RunModel(args)
    start_time = time.time()
    if not args.inference_mode:
        run_model.train()
    else:
        run_model.test()
    print("--- %s seconds ---" % (time.time() - start_time))