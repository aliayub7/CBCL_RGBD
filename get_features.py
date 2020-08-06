import numpy as np
import sys
import os
import time
import pickle
import cv2
from img_to_vec import Img2Vec
from PIL import Image

path_to_train = './data/SUN/training_set'
path_to_test = './data/SUN/testing_set'

categories = {'bathroom': 0, 'bedroom': 1, 'classroom': 2, 'computer_room': 3, 'conference_room': 4,
'corridor': 5, 'dining_area': 6, 'dining_room': 7, 'discussion_area': 8, 'furniture_store': 9,
 'home_office': 10, 'kitchen': 11, 'lab': 12, 'lecture_theatre': 13, 'library': 14, 'living_room': 15, 'office': 16, 'rest_space': 17, 'study_space': 18}

total_classes=19
total_num=[0]*total_classes

img2vec = Img2Vec(cuda = True, model = 'resnet_places')
train_features = []
train_labels = []
test_features = []
test_labels = []
import types
iter = 0
for label in os.listdir(path_to_train):
    folder = os.path.join(path_to_train,label)
    if label not in categories:
        categories[label] = categories['others']
    for file in os.listdir(folder):
        path = os.path.join(folder,file)
        img = Image.open(path)
        vec = img2vec.get_vec(img)
        if vec is not None:
            print ('iter',iter)
            iter+=1
            features_np=np.array(vec)
            features_f = features_np.flatten()
            train_features.append(features_f)
            train_labels.append(categories[label])
            total_num [categories[label]] += 1

iter = 0
for label in os.listdir(path_to_test):
    folder = os.path.join(path_to_test,label)
    if label not in categories:
        categories[label] = categories['others']
    for file in os.listdir(folder):
        path = os.path.join(folder,file)
        img = Image.open(path)
        vec = img2vec.get_vec(img)
        if vec is not None:
            print ('iter',iter)
            iter+=1
            features_np=np.array(vec)
            features_f = features_np.flatten()
            test_features.append(features_f)
            test_labels.append(categories[label])

print (total_num)

print (len(train_features))
train_features = np.array(train_features)
train_labels = np.array(train_labels)
test_features = np.array(test_features)
test_labels = np.array(test_labels)

with open('./features/train_SUN_rgb.data', 'wb') as filehandle:
    pickle.dump(train_features, filehandle)
with open('./features/test_SUN_rgb.data', 'wb') as filehandle:
    pickle.dump(test_features, filehandle)
with open('./features/labels_train_SUN_rgb.data', 'wb') as filehandle:
    pickle.dump(train_labels, filehandle)
with open('./features/labels_test_SUN_rgb.data', 'wb') as filehandle:
    pickle.dump(test_labels, filehandle)
