# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 2020

@author: Ali Ayub
"""

import numpy as np
from copy import deepcopy
import math
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from Functions_new import get_centroids
from Functions_new import get_validation_accuracy
from sklearn.model_selection import KFold
import random
# THE FOLLOWING IS DEFINITELY NEEDED WHEN WORKING WITH PYTORCH
import os
import time
#os.environ["OMP_NUM_THREADS"] = "1"


class getCentroids:
    def __init__(self,x_train,y_train,classes,seed,centroids_limit=None,current_centroids=[],increment=0,distance_metric='euclidean',clustering_type='Agglomerative_variant',
    k_base=1,k_limit=25,x_val=None,y_val=None,get_covariances=False,complete_covariances=[],complete_centroids_num=[],d_base=17.0,d_limit=23.0,d_step=0.2,
    diag_covariances=False,x_test=None,y_test=None,lower_centroids=[]):
        self.x_train = x_train
        self.y_train = y_train
        self.total_classes = classes
        self.increment = increment
        self.total_centroids_limit = centroids_limit
        self.complete_centroids = current_centroids
        self.distance_metric = distance_metric
        self.clustering_type = clustering_type
        self.k_base = k_base
        self.k_limit = k_limit
        self.best_k = None
        self.best_d = None
        self.total_num = []
        self.x_val = x_val
        self.y_val = y_val
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.get_covariances = get_covariances
        self.diag_covariances = diag_covariances
        self.complete_covariances = complete_covariances
        self.complete_centroids_num = complete_centroids_num
        self.d_base = d_base
        self.d_limit = d_limit
        self.d_step = d_step
        self.x_test = x_test
        self.y_test = y_test
        self.lower_centroids = lower_centroids

    def initialize(self,x_train,y_train,classes,seed,centroids_limit=None,current_centroids=[],increment=0,distance_metric='euclidean',clustering_type='Agglomerative_variant',
    k_base=1,k_limit=25,x_val=None,y_val=None,get_covariances=False,complete_covariances=[],complete_centroids_num=[],d_base=17.0,d_limit=23.0,d_step=0.2,
    diag_covariances=False,x_test=None,y_test=None,lower_centroids=[]):
        self.x_train = x_train
        self.y_train = y_train
        self.total_classes = classes
        self.increment = increment
        self.total_centroids_limit = centroids_limit
        self.complete_centroids = current_centroids
        self.distance_metric = distance_metric
        self.clustering_type = clustering_type
        self.k_base = k_base
        self.k_limit = k_limit
        self.best_k = None
        self.best_d = None
        self.x_val = x_val
        self.y_val = y_val
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.get_covariances = get_covariances
        self.diag_covariances = diag_covariances
        self.complete_covariances = complete_covariances
        self.complete_centroids_num = complete_centroids_num
        self.d_base = d_base
        self.d_limit = d_limit
        self.d_step = d_step
        self.x_test = x_test
        self.y_test = y_test
        self.lower_centroids = lower_centroids

    # NEEDS FIXING
    def validation_based(self):
        current_total_centroids = 0
        for i in range(0,len(self.complete_centroids)):
            current_total_centroids+=len(self.complete_centroids[i])

        if self.x_test is None:
            x_train_temp,x_val,y_train_temp,y_val = train_test_split(self.x_train,self.y_train,test_size=0.2,stratify = self.y_train)
        else:
            x_train_temp = self.x_train
            y_train_temp = self.y_train
            x_val = self.x_test
            y_val = self.y_test
        train_data = [[] for y in range(self.total_classes)]
        train_depth = [[] for y in range(self.total_classes)]
        total_num = deepcopy(self.total_num)
        total_num.extend([0 for y in range(self.total_classes)])
        for i in range(0,len(y_train_temp)):
            train_data[y_train_temp[i]-(self.increment*self.total_classes)].append(x_train_temp[i])
            total_num[y_train_temp[i]]+=1
        print ("total images per class:",total_num)

        weighting = np.divide([1 for x in range(0,(len(total_num)))],total_num)
        weighting = np.divide(weighting,np.sum(weighting))

        d_thresholds = []
        max_acs=[]
        indis = []
        # FIGURE OUT DISTANCE THRESHOLD
        #for distance_threshold in range(20,40,5):#range(55,120,5):#np.arange(13,21,0.5):#
        #for distance_threshold in range(20,28,1):
        #for distance_threshold in np.arange(30,30.5,0.5):
        for distance_threshold in np.arange(self.d_base,self.d_limit,self.d_step):
            d_thresholds.append(distance_threshold)

            # Make a training pack
            train_pack = []
            for i in range(0,self.total_classes):
                train_pack.append([train_data[i],train_depth[i],distance_threshold,w_dep])

            # create training data pack for multiprocessing
            train_pack = []
            for i in range(0,self.total_classes):
                train_pack.append([train_data[i],train_depth[i],distance_threshold,self.clustering_type,w_dep,None,None])

            # multiprocess for each class separately
            my_pool = Pool(total_classes)
            centroids_pack = my_pool.map(get_centroids,train_pack)
            my_pool.close()
            # unpack the centroids
            for i in range(0,total_classes):
                centroids[i] = centroids_pack[i][0]
                centroids_depth[i] = centroids_pack[i][1]

            temp_complete_centroids = deepcopy(self.complete_centroids)
            temp_complete_centroids.extend(centroids)

            temp_complete_centroids_depth = deepcopy(self.complete_centroids_depth)
            temp_complete_centroids_depth.extend(centroids_depth)


            val_pack = [x_val,x_val_depth,y_val,temp_complete_centroids,temp_complete_centroids_depth,self.k_limit,self.total_classes,self.increment,weighting]
            accuracies = get_validation_accuracy(val_pack)
            #accuracies = get_validation_accuracy_lower(val_pack)
            max_ac = np.max(accuracies)
            max_acs.append(max_ac)
            indi = np.argmax(accuracies) + self.k_base
            indis.append(indi)
            print ('distance threshold: {}, tops: {}, max_accuracy: {}'.format(distance_threshold,indi,max_ac))
        print ("distance_threshold: {}, tops: {}, max_accuracy: {}".format(d_thresholds[np.argmax(max_acs)],indis[np.argmax(max_acs)],max(max_acs)))

        # get the best centroids
        #centroids = [[[0 for x in range(len(x_train[0]))]] for y in range(total_classes)]
        self.total_num.extend([0 for y in range(self.total_classes)])
        train_data = [[] for y in range(self.total_classes)]
        for i in range(0,len(self.y_train)):
            train_data[self.y_train[i]-(self.increment*self.total_classes)].append(self.x_train[i])
            self.total_num[self.y_train[i]]+=1

        train_pack = []
        for i in range(0,self.total_classes):
            train_pack.append([train_data[i],d_thresholds[np.argmax(max_acs)],self.clustering_type,self.get_covariances,self.diag_covariances])
        if self.get_covariances!=True:
            my_pool = Pool(self.total_classes)
            centroids = my_pool.map(get_centroids,train_pack)
            my_pool.close()
        else:
            my_pool = Pool(self.total_classes)
            centroids_variances = my_pool.map(get_centroids,train_pack)
            my_pool.close()
            centroids = []
            covariances = []
            centroids_num = []
            for j in range(0,len(centroids_variances)):
                centroids.append(centroids_variances[j][0])
                covariances.append(centroids_variances[j][1])
                centroids_num.append(centroids_variances[j][2])

        exp_centroids = 0
        for i in range(0,len(centroids)):
            exp_centroids+=len(centroids[i])

        # reduce previous centroids if more than allowed, THIS HAS TO BE CHANGED FOR COVARIANCES
        if self.total_centroids_limit!=None:
            self.complete_centroids,self.complete_covariances,self.complete_centroids_num = check_reduce_centroids_covariances(self.complete_centroids,self.complete_covariances,self.complete_centroids_num,
            current_total_centroids,exp_centroids,self.total_centroids_limit,self.increment,
            self.total_classes)
        # add the new centroids to the complete_centroids
        self.complete_centroids.extend(centroids)
        if self.get_covariances==True:
            self.complete_covariances.extend(covariances)
            self.complete_centroids_num.extend(centroids_num)

        print ("total_classes",len(self.complete_centroids))
        total_centroids = 0
        for i in range(0,len(self.complete_centroids)):
            total_centroids+=len(self.complete_centroids[i])
        print ("total_centroids",total_centroids)

        self.best_k = indis[np.argmax(max_acs)]
        self.best_d = d_thresholds[np.argmax(max_acs)]

    # THIS HAS NOT BEEN updated for covariances and ohter things. TOO MANY UPDATES NEEDED
    def cross_validation_based(self):
        current_total_centroids = 0
        for i in range(0,len(self.complete_centroids)):
            current_total_centroids+=len(self.complete_centroids[i])

        folds = 5
        kf = KFold(n_splits=folds,shuffle=True)
        kf.get_n_splits(x_train)

        d_thresholds = []
        max_acs=[]
        indis = []
        for distance_threshold in range(30,220,5):#np.arange(0.5,30,1):#range(50,135,5):
            d_thresholds.append(distance_threshold)
            all_fold_accuracies = [0 for x in range(0,self.k_limit)]
            for train_index, test_index in kf.split(x_train):
                x_train_temp, x_val = [x_train[x] for x in train_index], [x_train[x] for x in test_index]
                y_train_temp, y_val = [y_train[x] for x in train_index], [y_train[x] for x in test_index]

                train_data = [[] for y in range(self.total_classes)]
                total_num = [0 for y in range(self.total_classes)]
                for i in range(0,len(y_train_temp)):
                    train_data[y_train_temp[i]-(self.increment*self.total_classes)].append(x_train_temp[i])
                    total_num[y_train[i]-(increment)*self.total_classes]+=1

                centroids = [[[0 for x in range(len(x_train[0]))]] for y in range(self.total_classes)]
                train_pack = []
                for i in range(0,self.total_classes):
                    train_pack.append([train_data[i],distance_threshold,self.clustering_type])

                weighting = np.divide([1 for x in range(0,(len(total_num)))],total_num)
                weighting = np.divide(weighting,np.sum(weighting))

                # multiprocess for each class separately
                my_pool = Pool(self.total_classes)
                centroids = my_pool.map(get_centroids,train_pack)
                my_pool.close()

                temp_complete_centroids = deepcopy(self.complete_centroids)
                temp_complete_centroids = check_reduce_centroids_covariances(temp_complete_centroids,current_total_centroids,temp_exp_centroids,self.total_centroids_limit,
                self.increment,self.total_classes)
                temp_complete_centroids.extend(centroids)

                # make test packs for each k-value
                val_pack = []
                for k in range(self.k_base,self.k_limit+self.k_base):
                    val_pack.append([x_val,y_val,temp_complete_centroids,k,self.total_classes,self.increment,weighting])
                my_pool = Pool(k_limit)
                accuracies = my_pool.map(get_accuracy,val_pack)
                my_pool.close()
                all_fold_accuracies = np.add(all_fold_accuracies,accuracies)
                #print ("time_passed",toc-tic)

            all_fold_accuracies = np.divide(all_fold_accuracies,folds)
            max_ac = np.max(all_fold_accuracies)
            max_acs.append(max_ac)
            indi = np.argmax(all_fold_accuracies) + 1
            indis.append(indi)
        print ("distance_threshold: {}, tops: {}, max_accuracy: {}".format(d_thresholds[np.argmax(max_acs)],indis[np.argmax(max_acs)],max(max_acs)))

        train_data = [[] for y in range(self.total_classes)]
        for i in range(0,len(self.y_train)):
            train_data[self.y_train[i]-(increment*self.total_classes)].append(self.x_train[i])

        # get the best centroids
        centroids = [[[0 for x in range(len(x_train_increment[0]))]] for y in range(total_classes)]

        train_pack = []
        for i in range(0,self.total_classes):
            train_pack.append([train_data[i],d_thresholds[np.argmax(max_acs)],self.clustering_type])

        my_pool = Pool(self.total_classes)
        centroids = my_pool.map(get_centroids,train_pack)
        my_pool.close()
        exp_centroids = 0
        for i in range(0,len(centroids)):
            exp_centroids+=len(centroids[i])

        # reduce previous centroids if more than allowed
        self.complete_centroids = check_reduce_centroids_covariances(self.complete_centroids,current_total_centroids,exp_centroids,self.total_centroids_limit,self.increment,
        self.total_classes)
        # add the new centroids to the complete_centroids
        self.complete_centroids.extend(centroids)

        # labels for centroids will be needed when classes are shuffled
        print ("total_classes",len(self.complete_centroids))
        total_centroids = 0
        for i in range(0,len(self.complete_centroids)):
            total_centroids+=len(self.complete_centroids[i])
        print ("total_centroids",total_centroids)

        self.best_k = indis[np.argmax(max_acs)]

    def validation_based_centroids(self):
        current_total_centroids = 0
        for i in range(0,len(self.complete_centroids)):
            current_total_centroids+=len(self.complete_centroids[i])

        x_train_temp,x_val,y_train_temp,y_val = train_test_split(self.x_train,self.y_train,test_size=0.2,stratify = self.y_train)
        train_data = [[] for y in range(self.total_classes+(self.increment*self.total_classes))]
        #total_num = [0 for y in range(self.total_classes)]
        total_num = deepcopy(self.total_num)
        total_num.extend([0 for y in range(self.total_classes+(self.increment*self.total_classes))])
        for i in range(0,len(y_train_temp)):
            train_data[y_train_temp[i]].append(x_train_temp[i])
            total_num[y_train_temp[i]]+=1
        #print ("total images per class:",total_num)

        weighting = np.divide([1 for x in range(0,(len(total_num)))],total_num)
        weighting = np.divide(weighting,np.sum(weighting))

        d_thresholds = []
        max_acs=[]
        indis = []
        # FIGURE OUT DISTANCE THRESHOLD
        for distance_threshold in range(10,50,5):#np.arange(13,21,0.5):#range(80,155,5):
            d_thresholds.append(distance_threshold)
            #centroids = [[[0 for x in range(len(x_train_temp[0]))]] for y in range(self.total_classes)]

            # create training data pack for multiprocessing
            train_pack = []
            for i in range(0,self.total_classes+(self.increment*self.total_classes)):
                train_pack.append([train_data[i],distance_threshold,self.clustering_type])
            # multiprocess for each class separately
            my_pool = Pool(self.total_classes)
            centroids = my_pool.map(get_centroids,train_pack)
            my_pool.close()

            temp_exp_centroids = 0
            for i in range(0,len(centroids)):
                temp_exp_centroids+=len(centroids[i])

            # temporarily reduce previous centroids if more than allowed
            temp_complete_centroids = deepcopy(self.complete_centroids)
            temp_complete_centroids = check_reduce_centroids(temp_complete_centroids,current_total_centroids,temp_exp_centroids,self.total_centroids_limit,
            self.increment,self.total_classes)
            temp_complete_centroids.extend(centroids)

            val_pack = []
            for k in range(self.k_base,self.k_limit + self.k_base):
                val_pack.append([x_val,y_val,temp_complete_centroids,k,self.total_classes+(self.increment*self.total_classes),weighting])
            my_pool = Pool(self.k_limit)
            accuracies = my_pool.map(get_test_accuracy,val_pack)
            my_pool.close()
            max_ac = np.max(accuracies)
            max_acs.append(max_ac)
            indi = np.argmax(accuracies) + self.k_base
            indis.append(indi)
            print ('distance threshold: {}, tops: {}, max_accuracy: {}'.format(distance_threshold,indi,max_ac))
        print ("distance_threshold: {}, tops: {}, max_accuracy: {}".format(d_thresholds[np.argmax(max_acs)],indis[np.argmax(max_acs)],max(max_acs)))

        # get the best centroids
        #centroids = [[[0 for x in range(len(x_train[0]))]] for y in range(total_classes)]
        self.total_num.extend([0 for y in range(self.total_classes+(self.increment*self.total_classes))])
        train_data = [[] for y in range(self.total_classes+(self.increment*self.total_classes))]
        for i in range(0,len(self.y_train)):
            train_data[self.y_train[i]].append(self.x_train[i])
            self.total_num[self.y_train[i]]+=1

        train_pack = []
        for i in range(0,self.total_classes+(self.increment*self.total_classes)):
            train_pack.append([train_data[i],d_thresholds[np.argmax(max_acs)],self.clustering_type])

        my_pool = Pool(self.total_classes)
        centroids = my_pool.map(get_centroids,train_pack)
        my_pool.close()
        exp_centroids = 0
        for i in range(0,len(centroids)):
            exp_centroids+=len(centroids[i])

        # reduce previous centroids if more than allowed
        self.complete_centroids = check_reduce_centroids(self.complete_centroids,current_total_centroids,exp_centroids,self.total_centroids_limit,self.increment,
        self.total_classes)
        # add the new centroids to the complete_centroids
        self.complete_centroids.extend(centroids)

        print ("total_classes",len(self.complete_centroids))
        total_centroids = 0
        for i in range(0,len(self.complete_centroids)):
            total_centroids+=len(self.complete_centroids[i])
        print ("total_centroids",total_centroids)

        self.best_k = indis[np.argmax(max_acs)]
