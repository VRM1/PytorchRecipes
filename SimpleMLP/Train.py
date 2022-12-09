from argparse import ArgumentParser
import torch
from Utils import initialize_arguments
import torch.nn as nn
from Model import SimpleLenet
import os
from Dataset import DataRepo
from Utils import EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score, roc_auc_score
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import mlflow.pytorch
from mlflow import MlflowClient


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
        self.d_name = args.dataset
        self.lr = args.learning_rate
        self.resume = args.resume
        self.start_epoch = 0
        # path to write trained weights
        self.train_weight_path = 'trained_weights/' + self.m_name + '-' + self.d_name + '-' + str(self.epochs) + \
                                 '-' + str(self.tr_b_sz) + '.pth'
        data_repo = DataRepo()
        self.n_classes, self.i_channel, self.i_dim, self.train_len, \
             self.valid_len, self.test_len, self.train_loader, \
                 self.valid_loader, self.test_loader = \
                     data_repo(args, self.d_name, True, \
                         self.tr_b_sz, self.tst_b_sz)
        
        if self.n_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.init_model()


    def init_model(self, load_weights=False, res_round=None):

        if self.m_name == 'lenet300-100':
            self.model = SimpleLenet(self.i_dim, self.n_classes)
        
        early_stop_callback = EarlyStopping(monitor="val_loss", \
             min_delta=0.01, patience=self.args.patience, verbose=False, mode="min")
        
        if DEVICE == 'gpu':
            self.trainer = pl.Trainer(accelerator=DEVICE, max_epochs=args.epochs, \
                 min_epochs=1, callbacks=[early_stop_callback])
        else:
            self.trainer = pl.Trainer(accelerator=DEVICE, max_epochs=args.epochs, \
                 min_epochs=1, callbacks=[early_stop_callback])
        # Auto log all MLflow entities
        mlflow.pytorch.autolog()

    def __load_pre_train_model(self):

        # get the name/path of the weight to be loaded
        self.getTrainedmodel()
        # load the weights
        if DEVICE.type == 'cpu':
            state = torch.load(self.train_weight_path, map_location=torch.device('cpu'))
        else:
            state = torch.load(self.train_weight_path)

        self.init_optimizer(self.lr)
        self.model.load_state_dict(state['weights'])
        self.start_epoch = state['epoch']
        # self.optimizer.load_state_dict(state['optimizer'])

    
    def train(self, experiment_id):
        
        with mlflow.start_run(experiment_id=experiment_id):
            if args.ckpt_path != 'None':
                self. trainer.fit(max_epochs=100, min_epochs=1, model=self.model, train_dataloaders=self.train_loader, \
                        val_dataloaders=self.valid_loader, ckpt_path=args.ckpt_path)
            else:
                self.trainer.fit(model=self.model, train_dataloaders=self.train_loader, \
                    val_dataloaders=self.valid_loader)
    
    def test(self, load_best_model=False):

        self.trainer.test(self.model, dataloaders=self.test_loader)
        preds  = self.trainer.predict(self.model, self.test_loader)
        y = torch.concat([p[1] for p in preds]).numpy()
        preds = torch.concat([p[0] for p in preds])
        preds = torch.nn.functional.softmax(preds).numpy()
        print(classification_report(y, preds.argmax(axis=1)))
        print("ROC-AUC:{}".format(roc_auc_score(y, preds[:, 1])))
        print("PrecisionRecall-AUC:{}".format(average_precision_score(y, preds[:, 1])))

    def getTrainedmodel(self):
        retrain = 100
        if self.is_bayesian:
            net_typ = '_is_bayesian_1'
        else:
            net_typ = '_is_bayesian_0'
        self.train_weight_path = 'trained_weights/' + self.m_name + '-' + self.d_name \
                                 + '-b' + str(self.tr_b_sz) + '-mcmc' + str(self.n_samples) + '-' \
                                 + net_typ + '-' + self.optim + '.pkl'
        return (self.model, self.train_weight_path)



if __name__ == '__main__':
    seed_everything(42)
    client = MlflowClient()
    try:
        experiment_id = client.create_experiment("PytorchRun")
    except:
        experiment_id = client.get_experiment_by_name("PytorchRun").experiment_id
    

    parser = ArgumentParser(description='')
    args = initialize_arguments(parser)

    
    run_model = RunModel(args)
    # write_summary = LogSummary(name=args.model + '_ba' + str(int(args.is_bayesian)) + '_' + args.dataset + '_' +
    #                                 str(args.b_sz) + '_' + str(args.epochs))
    
    run_model.train(experiment_id)
    run_model.test()