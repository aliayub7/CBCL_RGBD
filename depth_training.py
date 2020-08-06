import numpy as np
import sys
import os
import time
import pickle
from PIL import Image
from copy import deepcopy
import cv2
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import models, datasets

from training_functions import train
from training_functions import eval_training
from get_transformed_data import getTransformedData
from img_to_vec import Img2Vec
import random

seed=seed = random.randint(0,1000)
np.random.seed(seed)
torch.manual_seed(seed)

if __name__ == '__main__':

    path_to_train = './data/SUN/training_depth'
    path_to_test = './data/SUN/testing_depth'
    total_classes = 19

    # hyperparameters
    weight_decay = 5e-4
    classify_lr = 0.01
    classification_epochs = 200
    batch_size = 128

    #classify_net
    classify_net = models.vgg16()
    classify_net.classifier[6] = nn.Linear(in_features = 4096, out_features = total_classes)
    classify_net.load_state_dict(torch.load("./checkpoint/best_after"+str(classification_epochs)+"NYU"))
    #classify_net.fc = nn.Linear(in_features = 4096, out_features = total_classes)


    # loss functions
    loss_classify = nn.CrossEntropyLoss()

    # SUN Depth
    mean = [0.6983, 0.3918, 0.4474]
    std = [0.1648, 0.1359, 0.1644]
    #NYU Depth
    #mean = [0.4951, 0.3601, 0.4587]
    #std = [0.1474, 0.1950, 0.1646]

    # define transforms
    transforms_classification_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    transforms_classification_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])


    # classifier training
    train_dataset_classification = datasets.ImageFolder(path_to_train,transforms_classification_train)
    test_dataset_classification = datasets.ImageFolder(path_to_test,transforms_classification_test)

    dataloaders_train_classification = torch.utils.data.DataLoader(train_dataset_classification,batch_size = batch_size,
    shuffle=True, num_workers = 4)
    dataloaders_test_classification = torch.utils.data.DataLoader(test_dataset_classification,batch_size = batch_size,
    shuffle=True, num_workers = 4)

    optimizer = optim.SGD(classify_net.parameters(),lr=classify_lr,weight_decay=weight_decay,momentum=0.9)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60,120,160], gamma=0.2) #learning rate decay
    classify_net = classify_net.cuda()

    epoch_acc = eval_training(classify_net,dataloaders_test_classification,loss_classify,seed)

    # now classification phase
    since=time.time()
    for epoch in range(1, classification_epochs):
        train_scheduler.step(epoch)
        classification_loss = train(classify_net,dataloaders_train_classification,optimizer,loss_classify)
        print ('epoch:', epoch, '  classification loss:', classification_loss, '  learning rate:', optimizer.param_groups[0]['lr'])
        epoch_acc = eval_training(classify_net,dataloaders_test_classification,loss_classify,seed)
        print (' ')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    torch.save(classify_net.state_dict(), "./checkpoint/best_after"+str(classification_epochs)+"SUNRGBD")
