'''
In this program, I use the trained embeddings from Doc2Vec model and train a simple 2 layer NN for classifying good and bad patents
'''
import argparse
import torch
from Model import Lenet300_100
import torch.nn as nn
from Utils import AmazonReviewDataset
from sklearn.metrics import accuracy_score, roc_auc_score
from Dataset import old_breastCancerDataset
import torch.nn.functional as F
'''
Resources:
1. https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/pytorch-mnist-mlp.ipynb
'''


def getModel(i_dim, o_dim, h_dim, d_rate, device):
    print('Using PyTorch version:', torch.__version__, ' Device:', device)
    model = Lenet300_100(i_dim, o_dim, d_rate).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    optimizer = torch.optim.RMSprop(model.parameters(), lr = 1e-3)
    criterion = nn.CrossEntropyLoss()
    print(model)
    return (model, optimizer, criterion)


def Train(model, criterion, optimizer, patent_data, epoch, device):
    # model training: whole dataset at a time

    for epoch in range(epoch):
        for i, (x, y) in enumerate(patent_data):
            x, y = x.float().to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print("Epoch: {}, Loss: {:.5f}".format(epoch + 1, loss.item()))


def Test(model,test_loader,device):
    y_true, y_prob, y_pred = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.float().to(device)
            outputs = F.softmax(model(x))
            y_true += list(y.cpu().numpy())
            y_prob += list(outputs.cpu().numpy()[:, -1])

            # predicted label
            _, predicted = torch.max(outputs.data, 1)
            predicted = list(predicted.cpu().numpy())
            y_pred += predicted
    accuracy = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)
    return (accuracy,auc_roc)

# method to do K-fold using conventional pytorch method
def Main(data_name):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # data = AmazonReviewDataset(pth, fil_a, fil_b)
    if data_name == 'breast_cancer':
        data = old_breastCancerDataset()
    i_dim, o_dim = data.shape[-1], 2
    h_dim, d_rate = 100, 0.2
    epoch = 100
    
    num_instances = len(data)
    test_ratio = 0.4
    fold = 5
    test_size = int(num_instances * test_ratio)
    train_size = num_instances - test_size
    accuracies, aucs = [], []
    for k in range(fold):
        print('fold:{}'.format(k))
        train_data, test_data = torch.utils.data.random_split(data, (train_size, test_size))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
        model, optimizer, criterion = getModel(i_dim, o_dim, h_dim, d_rate, device)
        Train(model, criterion, optimizer, train_loader, epoch, device)
        acc, auc = Test(model,test_loader,device)
        accuracies.append(acc)
        aucs.append(auc)

    print('accuracies for {} folds:{}'.format(k,accuracies))
    print('AUCs for {} folds:{}'.format(k,aucs))


if __name__ == '__main__':
    # Note: amazon dataset does not work - need to fix

    parser = argparse.ArgumentParser('')
    parser.add_argument('-d','--dataset',help='datasets 1. breast_cancer 2. amazon_reviews', default='breast_cancer')
    args = parser.parse_args()
    pth = '/home/vineeth/Documents/DataRepo/AmazonReviews/Musical_Instrument/'
    fil_a = 'Musical_Instrument_reviews.d2v'
    fil_b = 'reviews_Musical_Instruments.json'
    Main(args.dataset)
