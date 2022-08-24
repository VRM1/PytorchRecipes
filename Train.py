'''
This program trains the following self.models with Cifar-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html

1. Resnet-5
'''
import yaml
from argparse import ArgumentParser, Namespace
import torch
import torch.nn as nn
from Model import SimpleLenet
import os
from torchinfo import summary
from Dataset import DataRepo
from tqdm import tqdm
from Utils import EarlyStopping
from Utils import LogSummary
import yaml
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import mlflow.pytorch
from mlflow import MlflowClient


if torch.cuda.is_available():
    DEVICE = 'gpu'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

'''
This program uses CIFAR10 data: https://www.cs.toronto.edu/~kriz/cifar.html for image classification using
several popular self.models based on convolution neural network.
'''
if not os.path.isdir('trained_weights'):
    os.makedirs('trained_weights')


class RunModel:

    def __init__(self, args):

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
             min_delta=0.01, patience=3, verbose=False, mode="min")
        
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
    
    def test_v2(self, load_best_model=False):

        self.trainer.test(self.model, dataloaders=self.test_loader)

    def test(self, is_valid=False, load_best_model=False):

        if load_best_model or self.test_mode:
            self.__load_pre_train_model()
        correct = 0
        total = 0
        if is_valid:
            data = self.valid_loader
        else:
            data = self.test_loader
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(tqdm(data)):
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                outputs = self.model(X)

                if self.n_classes != 1:
                    _, predicted = outputs.max(1)
                else:
                    outputs = torch.sigmoid(outputs.view(-1))
                    predicted = (outputs > 0.5).float()
                total += Y.size(0)
                correct += predicted.eq(Y).sum().item()
        t_accuracy = (100. * correct / total)
        return t_accuracy

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

def _initialize_arguments(parser):

    parser.add_argument('-config', help='configuration file *.yml', \
         type=str, required=False, default='None')
    parser.add_argument('-m', '--model', help='model name 1.lenet300-100', \
         default='lenet300-100')
    parser.add_argument('-test', '--test', help='if you want to run in test mode', \
         action='store_true')
    parser.add_argument('-b', '--b_sz', help='batch size', default=256, type=int)
    parser.add_argument('-dataset', help='datasets 1. breast_cancer 2. \
         covid19 3. long_document',default='breast_cancer')
    parser.add_argument('-data_path', help='the complete path of data', required=False)
    parser.add_argument('-e', '--epochs', help='number of epochs', default=150, type=int)
    parser.add_argument('-lr', '--learning_rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-op', '--optimizer', help='optimizer types, 1. SGD 2. Adam, \
         default SGD', default='Adam')
    parser.add_argument('-ba', '--is_bayesian', help='to use bayesian \
         layer or not', action='store_true')
    parser.add_argument('-is_valid', help='user validation data or not: 1 or 0 respectively', \
         type=int, default=0)
    parser.add_argument('-r', '--resume', help='if you want to resume from an epoch', \
         action='store_true')
    parser.add_argument('-patience', help='for early stopping. How many epochs to wait', \
         default=10, type=int)
    parser.add_argument('-report_test', help='if you want test the model at every training \
         epoch (disabling this will reduce moel training time)', action='store_true')
    parser.add_argument('-ckpt_path', help='Path to the checkpoint file, if you want \
         to load the pre-trained state of the model', required=False, default='None', type=str)
    parser.add_argument('-req_feattures', help='required features', required=False, type=str)
    parser.add_argument('-target_label', help='name of the target feature column', \
         required=False, type=str)
    args = parser.parse_args()
    if args.config != 'None':
        opt = vars(args)
        args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(args)
        args = Namespace(**opt)
    return args

if __name__ == '__main__':
    seed_everything(42)
    client = MlflowClient()
    try:
        experiment_id = client.create_experiment("PytorchRun")
    except:
        experiment_id = client.get_experiment_by_name("PytorchRun").experiment_id
    

    parser = ArgumentParser(description='')
    args = _initialize_arguments(parser)

    
    run_model = RunModel(args)
    write_summary = LogSummary(name=args.model + '_ba' + str(int(args.is_bayesian)) + '_' + args.dataset + '_' +
                                    str(args.b_sz) + '_' + str(args.epochs))
    
    run_model.train(experiment_id)
    run_model.test_v2()