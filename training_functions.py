import torch
import torch.nn as nn
from torch.optim import SGD
from torch.autograd import Variable
from copy import deepcopy
import time
from tqdm import tqdm
import numpy as np
import random
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.optim as optim
m_lambda = 0.3

def train(classify_net,dataloaders_train_classification,optimizer,loss_classify,lambda_based=None,seed=1):
    #np.random.seed(seed)
    #random.seed(seed)
    classify_net.train()
    total_loss = []
    for images, labels in dataloaders_train_classification:
        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        #outputs,_,_ = classify_net(images)
        outputs = classify_net(images)

        loss = loss_classify(outputs, labels)
        loss.backward()
        total_loss.append(loss.item())
        optimizer.step()
    return np.average(total_loss)

def eval_training(classify_net,dataloaders_test_classification,loss_classify,seed=1):
    #np.random.seed(seed)
    #random.seed(seed)
    classify_net.eval()
    test_loss = 0.0 # cost function error
    correct = 0.0
    for images, labels in dataloaders_test_classification:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        #outputs,_,_ = classify_net(images)
        outputs = classify_net(images)
        loss = loss_classify(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(dataloaders_test_classification.dataset),
        correct.float() / len(dataloaders_test_classification.dataset)
    ))
    return correct.float() / len(dataloaders_test_classification.dataset)

def centroids_train(x_train,x_train_depth,y_train,model,criterion,optimizer,device,seed,batch_size = 64,
    centroids=None,centroids_depth=None,cent_labels=None,w_dep=None,k_limit= None,total_classes=19,weighting=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model.train()
    permutation = torch.randperm(x_train.size()[0])
    training_loss = []
    for i in range(0,x_train.size()[0],batch_size):
        indices = permutation[i:i+batch_size]
        batch_x,batch_y = x_train[indices],y_train[indices]
        batch_x_depth = x_train_depth[indices]
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x_depth = batch_x_depth.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x,batch_x_depth,centroids,centroids_depth,cent_labels,w_dep,k_limit,total_classes,weighting)
        loss = criterion(outputs, batch_y)
        loss = Variable(loss,requires_grad=True)
        loss.backward()
        training_loss.append(loss.item())
        optimizer.step()
    return np.average(training_loss)

def single_batch_train (x_train,y_train,model,criterion, optimizer,device,seed,batch_size = 64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model.train()
    permutation = torch.randperm(x_train.size()[0])
    training_loss = []
    for i in range(0,x_train.size()[0],batch_size):
        indices = permutation[i:i+batch_size]
        batch_x,batch_y = x_train[indices],y_train[indices]
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs,batch_y)

        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.average(training_loss)

def eval_model(x_test,y_test,model,criterion,device,batch_size = 64):
    model.eval()
    test_loss = [] # cost function error
    correct = 0.0
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    outputs = model(x_test)
    loss = criterion(outputs,y_test)
    test_loss.append(loss.item())
    _,preds = torch.max(outputs,1)
    correct += torch.sum(preds == y_test)

    test_loss = np.average(test_loss)
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss,
        correct.float() / len(y_test)
    ))
    return correct.float()/len(y_test)
