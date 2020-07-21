'''
This program trains the following self.models with Cifar-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html

1. Resnet-5
'''
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from Model import Lenet300_100
import os
import numpy as np
from torchsummary import summary
import GPUtil
import torch as th
from Dataset import DataRepo
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from Utils import EarlyStopping
import torch.nn.functional as F
import pdb

torch.manual_seed(33)
np.random.seed(33)
if torch.cuda.is_available():
    # check which gpu is free and assign that gpu
    AVAILABLE_GPU = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.5, \
                                        maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])[0]
    th.cuda.set_device(AVAILABLE_GPU)
    print('Program will be executed on GPU:{}'.format(AVAILABLE_GPU))
    DEVICE = torch.device('cuda:' + str(AVAILABLE_GPU))
else:
    DEVICE = torch.device('cpu')

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
        self.n_classes, self.i_channel, self.i_dim, self.train_len, self.valid_len,\
        self.test_len, self.train_loader, self.valid_loader, self.test_loader = data_repo(self.d_name, True, self.tr_b_sz, self.tst_b_sz)
        if self.n_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.init_model()
        self.init_optimizer(self.lr)

    def get_validation_data(self, is_valid):

        indices = range(self.train_len)
        split = int(np.floor(0.1 * self.train_len))
        valid_indx = np.random.choice(indices, split)
        train_indx = set(indices).difference(set(valid_indx))
        train_sampler = SubsetRandomSampler(list(train_indx))
        valid_sampler = SubsetRandomSampler(valid_indx)
        self.train_loader = DataLoader(self.train_d, batch_size=self.tr_b_sz, sampler=train_sampler, num_workers=1)
        self.valid_loader = DataLoader(self.train_d, batch_size=256, sampler=valid_sampler, num_workers=1)

    def init_model(self, load_weights=False, res_round=None):

        if self.m_name == 'lenet300-100':
            self.model = Lenet300_100(self.i_dim, self.n_classes).to(DEVICE)
            # torch.nn.DataParallel(self.model.features)
            t_param = sum(p.numel() for p in self.model.parameters())


        if self.resume:
            self.__load_pre_train_model()
        print('Running Mode:{}, #TrainingSamples:{}, #ValidationSamples:{}, #TestSamples:{}, #Parameters:{} ResumingFromEpoch:{}'
              .format(self.m_name, self.train_len, self.valid_len, self.test_len, t_param, self.start_epoch))

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

    def init_optimizer(self, l_rate=0.001):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=l_rate, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=l_rate)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100,150,200], gamma=0.1)

    def train(self):

        train_loss = []
        correct = 0
        total = 0
        self.model.train()
        # Decay Learning Rate
        self.scheduler.step()
        for batch_idx, (X, Y) in enumerate(tqdm(self.train_loader)):

            X, Y = X.to(DEVICE), Y.to(DEVICE)
            self.optimizer.zero_grad()
            if self.is_bayesian:
                loss = self.model.sample_elbo(inputs=X,
                                              labels=Y,
                                              criterion=self.criterion,
                                              sample_nbr=self.n_samples,
                                              complexity_cost_weight=1 / 50000)
            else:
                outputs = self.model(X)
                # loss = self.criterion(outputs.view(-1), Y)
                loss = self.criterion(outputs, Y)
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            if self.is_bayesian:
                outputs = self.model(X)

            if self.n_classes != 1:
                _, predicted = outputs.max(1)
            else:
                outputs = torch.sigmoid(outputs.view(-1))
                predicted = (outputs > 0.5).float()
            total += Y.size(0)
            correct += predicted.eq(Y).sum().item()

        t_accuracy = (100. * correct / total)
        avg_train_loss = np.average(train_loss)
        return avg_train_loss, t_accuracy

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--model', help='model name 1.lenet300-100', default='lenet300-100')
    parser.add_argument('-test','--test',help='if you want to run in test mode', action='store_true')
    parser.add_argument('-b', '--b_sz', help='batch size', default=256, type=int)
    parser.add_argument('-d','--dataset',help='datasets 1. breast_cancer 2. covid19 3. long_document', default='breast_cancer')
    parser.add_argument('-e', '--epochs', help='number of epochs', default=150, type=int)
    parser.add_argument('-lr', '--learning_rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-op', '--optimizer', help='optimizer types, 1. SGD 2. Adam, default SGD', default='Adam')
    parser.add_argument('-ba', '--is_bayesian', help='to use bayesian layer or not', action='store_true')
    parser.add_argument('-v', '--is_valid', help='whether to use validation or not', action='store_true')
    parser.add_argument('-r', '--resume', help='if you want to resume from an epoch', action='store_true')

    args = parser.parse_args()
    run_model = RunModel(args)

    if not args.test:
        patience = 5
        start_epoch = 0
        if args.resume:
            start_epoch = run_model.start_epoch

        early_stopping = EarlyStopping(patience=patience, verbose=True, typ='loss')
        for e in range(start_epoch, args.epochs):
            avg_train_loss, train_accuracy = run_model.train()
            if args.is_valid:
                valid_accuracy = run_model.test(is_valid=True)

            tst_accuracy = run_model.test()
            model, path_to_write = run_model.getTrainedmodel()
            early_stopping(e, avg_train_loss, model, run_model.optimizer, path_to_write)
            if early_stopping.early_stop:
                break
            if args.is_valid:
                print('Epoch:{}, Lr:{} AvgTrainLoss:{:.3f}, TrainAccuracy:{:.2f}, ValidationAccuracy:{:.2f}, TestAccuracy:{:.2f}'
                      .format(e, run_model.scheduler.get_lr(), avg_train_loss, train_accuracy, valid_accuracy, tst_accuracy))
            else:
                print('Epoch:{}, Lr:{}, AvgTrainLoss:{:.3f}, TrainAccuracy:{:.2f}, TestAccuracy:{:.2f}'
                      .format(e, run_model.scheduler.get_lr(), avg_train_loss, train_accuracy, tst_accuracy))

        tst_accuracy = run_model.test(load_best_model=True)
        print('Final test accuracy on best model:{}'.format(tst_accuracy))

    else:
        tst_accuracy = run_model.test()
        print('Final test accuracy on best model:{:.3f}'.format(tst_accuracy))
