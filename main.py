# -*- coding: utf-8 -*-
"""
Created on Thu Aug 6 2020

@author: Ali Ayub
"""

import numpy as np
from multiprocessing import Pool
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
import math
import time
import scipy
import random
import json

import seaborn as sns; sns.set()
import pandas as pd

from Functions_new import get_validation_accuracy
from Functions_new import get_centroids
from Functions_new import confusion_centroids
from Functions_new import get_silhouette
from Functions_new import get_silhouette_other
from Functions_new import predict_multiple_k

distance_metric = 'euclidean'
clustering_type = 'Agglomerative_variant'

with open('./features/train_SUN_rgb.data', 'rb') as filehandle:
    x_train = pickle.load(filehandle)
with open('./features/test_SUN_rgb.data', 'rb') as filehandle:
    x_test = pickle.load(filehandle)
with open('./features/labels_train_SUN_rgb.data', 'rb') as filehandle:
    y_train = pickle.load(filehandle)
with open('./features/labels_test_SUN_rgb.data', 'rb') as filehandle:
    y_test = pickle.load(filehandle)

with open('./features/train_SUN_depth.data', 'rb') as filehandle:
    x_train_depth = pickle.load(filehandle)
with open('./features/test_SUN_depth.data', 'rb') as filehandle:
    x_test_depth = pickle.load(filehandle)
with open('./features/labels_train_SUN_depth.data', 'rb') as filehandle:
    y_train_depth = pickle.load(filehandle)
with open('./features/labels_test_SUN_depth.data', 'rb') as filehandle:
    y_test_depth = pickle.load(filehandle)


w_rgb = 1.0
total_classes = 19

train_data = [[] for y in range(total_classes)]
total_num = [0 for y in range(total_classes)]
for i in range(0,len(y_train)):
    train_data[y_train[i]].append(x_train[i])
    total_num[y_train[i]]+=1

train_depth = [[] for y in range(total_classes)]
for i in range(0,len(y_train)):
    train_depth[y_train[i]].append(x_train_depth[i])

weighting = np.divide([1 for x in range(0,total_classes)],total_num)
weighting = np.divide(weighting,sum(weighting))

#0.87,140,3

w_dep = 0.73
distance_threshold = 85
centroids = [[[0 for x in range(len(x_train[0]))]] for y in range(total_classes)]
centroids_depth = [[[0 for x in range(len(x_train_depth[0]))]] for y in range(total_classes)]
image_indices = []

# Make a training pack
train_pack = []
for i in range(0,total_classes):
    train_pack.append([train_data[i],train_depth[i],distance_threshold,w_dep,clustering_type])

# multiprocess for each class separately
my_pool = Pool(total_classes)
centroids_pack = my_pool.map(get_centroids,train_pack)
my_pool.close()
# unpack the centroids
for i in range(0,total_classes):
    centroids[i] = centroids_pack[i][0]
    centroids_depth[i] = centroids_pack[i][1]
    image_indices.append(centroids_pack[i][2])

total_centroids = 0
for i in range(0,total_classes):
    total_centroids+=len(centroids[i])
#print ('total centroids',total_centroids)
bad_centroids = None

k_base = 17
k_limit = 17
test_pack = [x_test,x_test_depth,y_test,centroids,centroids_depth,w_dep,k_limit,weighting,total_classes,bad_centroids]
accuracies,_,_ = get_validation_accuracy(test_pack)
max_ac = np.max(accuracies)
print ("distance_threshold: {}, tops: {}, max_accuracy: {}".format(distance_threshold,k_base,accuracies))

### TO FIND THE CONFUSED CLASSES
train_pack = []
for i in range(0,total_classes):
    train_pack.append([train_data[i],train_depth[i],centroids,centroids_depth,w_dep,image_indices[i],i])

# multiprocess for each class separately
my_pool = Pool(total_classes)
return_pack = my_pool.map(get_silhouette_other,train_pack)
my_pool.close()
silhouttes = []
other_classes = []
for i in range(0,len(return_pack)):
    temp = []
    class_temp = []
    for j in range(len(return_pack[i][0])):
        temp.extend(return_pack[i][0][j])
        class_temp.extend(return_pack[i][1][j])
    silhouttes.append(temp)
    other_classes.append(class_temp)

occurences = [[0 for x in range(total_classes)] for y in range(0,total_classes)]
for which_class in range(0,total_classes):
    indis = [x for x in range(0,len(silhouttes[which_class])) if silhouttes[which_class][x]<=0.01]
    cls = np.array(other_classes[which_class])
    cls = cls[indis]
    cls = list(cls)

    for i in range(0,total_classes):
        occurences[which_class][i] += cls.count(i)
        occurences[i][which_class] += cls.count(i)
    print ('percentage',len(cls)/len(other_classes[which_class]))
for i in range(0,total_classes):
    print (occurences[i])

labels = [i for i in range(0,total_classes)]

plt.figure(figsize = (11,9))
sns.set(font_scale=0.8)#for label size
confusion = np.array(occurences)
sns.heatmap(confusion,annot=True,xticklabels=labels,yticklabels=labels,annot_kws={"size": 5})
plt.show()
