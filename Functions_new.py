"""
Created on 08/06/2020

@author: Ali Ayub
"""
# THIS FILE IS ONLY FOR INCREMENTAL LEARNING

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
from copy import deepcopy
import math
from multiprocessing import Pool
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fclusterdata
from scipy.cluster.hierarchy import fcluster, ward, average, weighted, complete, single
from scipy.spatial.distance import pdist
from multiprocessing import Pool
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"

distance_metric = 'euclidean'
w_rgb = 1.0

def my_dist(data1,data2):
    d_rgb = find_distance(data1[0],data2[0],distance_metric)
    d_dep = find_distance(data1[1],data2[1],distance_metric)
    dist = (w_rgb*d_rgb + data1[2]*d_dep)/2
    return dist

def get_centroids(train_pack):
    # unpack x_train and x_train_depth
    x_train = train_pack[0]
    x_train_depth = train_pack[1]
    distance_threshold = train_pack[2]
    w_dep = train_pack[3]
    clustering_type = train_pack[4]
    if clustering_type=='Agglomerative_variant':
        # for each training sample do the same stuff...
        centroids = [[0 for x in range(len(x_train[0]))]]
        centroids_depth = [[0 for x in range(len(x_train_depth[0]))]]
        image_indices = [[]]

        # initalize centroids
        centroids[0] = x_train[0]
        centroids_depth[0] = x_train_depth[0]
        total_num = [1]
        image_indices[0].append(0)
        for i in range(1,len(x_train)):
            distances=[]
            indices = []
            for j in range(0,len(centroids)):
                d_rgb = find_distance(x_train[i],centroids[j],distance_metric)
                d_dep = find_distance(x_train_depth[i],centroids_depth[j],distance_metric)
                d = (w_rgb*d_rgb + w_dep*d_dep)/2
                if d<distance_threshold:
                    distances.append(d)
                    indices.append(j)
            if len(distances)==0:
                centroids.append(x_train[i])
                centroids_depth.append(x_train_depth[i])
                total_num.append(1)
                image_indices.append([i])
            else:
                min_d = np.argmin(distances)
                centroids[indices[min_d]] = np.add(centroids[indices[min_d]],x_train[i])
                centroids_depth[indices[min_d]] = np.add(centroids_depth[indices[min_d]],x_train_depth[i])
                #centroids[indices[min_d]] = np.add(np.multiply(total_num[indices[min_d]],centroids[indices[min_d]]),x_train[i])
                #centroids_depth[indices[min_d]] = np.add(np.multiply(total_num[indices[min_d]],centroids_depth[indices[min_d]]),x_train_depth[i])
                total_num[indices[min_d]]+=1
                image_indices[indices[min_d]].append(i)
                #centroids[indices[min_d]] = np.divide(centroids[indices[min_d]],(total_num[indices[min_d]]))
                #centroids_depth[indices[min_d]] = np.divide(centroids_depth[indices[min_d]],total_num[indices[min_d]])
        for j in range(0,len(total_num)):
            centroids[j]=np.divide(centroids[j],total_num[j])
            centroids_depth[j]=np.divide(centroids_depth[j],total_num[j])
    elif clustering_type == 'k_means':
        kmeans = KMeans(n_clusters=distance_threshold, random_state = 0).fit(x_train)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        total_num = [0 for x in range(0,max(labels)+1)]
        per_labels = [[] for y in range(0,max(labels)+1)]
        for j in range(len(x_train)):
            per_labels[labels[j]].append(x_train[j])
            total_num[labels[j]]+=1
        covariances = [[] for y in range(0,max(labels)+1)]
        for j in range(0,max(labels)+1):
            if get_covariances==True:
                if diag_covariances != True:
                    covariances[j] = np.cov(np.array(per_labels[j]).T)
                else:
                    temp = np.cov(np.array(per_labels[j]).T)
                    covariances[j] = temp.diagonal()
    elif clustering_type == 'Agglomerative':
        train_data = []
        for i in range(0,len(x_train)):
            train_data.append([x_train[i],x_train_depth[i],w_dep])

        dist_mat=pdist(train_data,metric=my_dist)
        Z = linkage(dist_mat,'centroid')
        dn = hierarchy.dendrogram(Z)
        labels=fcluster(Z, t=distance_threshold, criterion='distance')

        total_number = [0 for x in range(0,max(labels))]
        centroids = [[0 for x in range(len(x_train[0]))] for y in range(0,max(labels))]
        centroids_depth = [[0 for x in range(len(x_train_depth[0]))] for y in range(0,max(labels))]
        for j in range(0,len(x_train)):
            centroids[labels[j]-1]+=x_train[j]
            centroids_depth[labels[j]-1]+=x_train_depth[j]
            total_number[labels[j]-1]+=1
        for j in range(0,len(centroids)):
            centroids[j] = np.divide(centroids[j],total_number[j])
            centroids_depth[j] = np.divide(centroids_depth[j],total_number[j])

    return [centroids,centroids_depth,image_indices]


def find_distance(data_vec,centroid,distance_metric):
    if distance_metric=='euclidean':
        return np.linalg.norm(data_vec-centroid)
    elif distance_metric == 'euclidean_squared':
        return np.square(np.linalg.norm(data_vec-centroid))
    elif distance_metric == 'cosine':
        return distance.cosine(data_vec,centroid)

def predict_multiple_class(pack):
    data_vec = pack[0]
    data_depth = pack[1]
    centroids = pack[2]
    centroids_depth = pack[3]
    class_centroid = pack[4]
    distance_metric = pack[5]
    w_dep = pack[6]
    bad_centroids=pack[7]
    if distance_metric=='euclidean':
        dists = np.subtract(data_vec,centroids)
        dists = np.linalg.norm(dists,axis=1)
        dists_dep = np.subtract(data_depth,centroids_depth)
        dists_dep = np.linalg.norm(dists_dep,axis=1)
    if bad_centroids is not None:
        dist = [[((w_rgb*dists[x])+(w_dep*dists_dep[x]))/2,class_centroid,bad_centroids[x]] for x in range(len(centroids))]
    else:
        dist = [[((w_rgb*dists[x])+(w_dep*dists_dep[x]))/2,class_centroid] for x in range(len(centroids))]
    return dist

def predict_multiple_k(data_vec,data_depth,centroids,centroids_depth,distance_metric,w_dep,tops,weighting,bad_centroids = None):
    dist = []
    for i in range(0,len(centroids)):
        if bad_centroids is not None:
            temp = predict_multiple_class([data_vec,data_depth,centroids[i],centroids_depth[i],i,distance_metric,w_dep,bad_centroids[i]])
        else:
            temp = predict_multiple_class([data_vec,data_depth,centroids[i],centroids_depth[i],i,distance_metric,w_dep,None])
        dist.extend(temp)
    sorted_dist = sorted(dist)
    common_classes = [0]*len(centroids)
    if tops>len(sorted_dist):
        tops = len(sorted_dist)
    # for all k values
    all_tops = []
    comm_alls = []
    found_bads = [0 for x in range(0,tops+1)]
    for k in range(0,tops+1):
        if k<len(sorted_dist):
            if bad_centroids is not None:
                if sorted_dist[k][2]>0:
                    found_bads[k] += 1
            if sorted_dist[k][0]==0.0:
                common_classes[sorted_dist[k][1]] += 1
            else:
                common_classes[sorted_dist[k][1]] += (1/sorted_dist[k][0])
            all_tops.append(np.argmax(np.multiply(common_classes,weighting)))
            comm_alls.append(np.multiply(common_classes,weighting))
        else:
            all_tops.append(-1)
    return all_tops,found_bads,comm_alls

def get_accu(pack):
    x_test = pack[0]
    x_test_depth = pack[1]
    y_test = pack[2]
    centroids = pack[3]
    centroids_depth = pack[4]
    distance_metric = pack[5]
    w_dep = pack[6]
    k = pack[7]
    weighting = pack[8]
    total_classes = pack[9]
    bad_centroids = pack[10]
    wrongs = 0.0
    bad_wrongs = 0.0

    #rhos = [[0 for x in range(0,len(x_test))] for y in range(k)]
    accus = [[0.0 for x in range(total_classes)] for y in range(k)]
    total_labels = [0.0 for x in range(total_classes)]
    com_alls = [[] for x in range(k)]
    for i in range(0,len(y_test)):
        total_labels[y_test[i]]+=1
        predicted_label,found_bads,temp=predict_multiple_k(x_test[i],x_test_depth[i],centroids,centroids_depth,distance_metric,w_dep,k,weighting,bad_centroids)
        for j in range(0,k):
            accus[j][y_test[i]]+=(predicted_label[j]==y_test[i])
            com_alls[j].append(temp[j])
            if bad_centroids is not None:
                wrongs += 1-(predicted_label[j]==y_test[i])
                bad_wrongs += (1-(predicted_label[j]==y_test[i]))*found_bads[j]
    return [accus,total_labels,wrongs,bad_wrongs,com_alls]

# get validation accuracy for all the k values
def get_validation_accuracy(test_pack):
    x_test = test_pack[0]
    x_test_depth = test_pack[1]
    y_test = test_pack[2]
    centroids = test_pack[3]
    centroids_depth = test_pack[4]
    w_dep = test_pack[5]
    k = test_pack[6]
    weighting = test_pack[7]
    total_classes = test_pack[8]
    bad_centroids = test_pack[9]

    accus = [[0.0 for x in range(total_classes)] for y in range(k)]
    total_labels = [0.0 for x in range(total_classes)]
    acc=0

    # divide y_test in 24 equal segments
    how_many = round(len(y_test)/24)
    pack = []
    now=0
    while now<len(y_test):
        if now+how_many>=len(y_test):
            pack.append([x_test[now:len(y_test)],x_test_depth[now:len(y_test)],y_test[now:len(y_test)],centroids,centroids_depth
            ,'euclidean',w_dep,k,weighting,total_classes,bad_centroids])
        else:
            pack.append([x_test[now:how_many+now],x_test_depth[now:how_many+now],y_test[now:how_many+now],centroids,centroids_depth
            ,'euclidean',w_dep,k,weighting,total_classes,bad_centroids])
        now+=how_many

    wrongs = 0
    bad_wrongs = 0
    my_pool = Pool(25)
    return_pack = my_pool.map(get_accu,pack)
    my_pool.close()
    com_alls = [[] for x in range(k)]
    for i in range(0,len(return_pack)):
        for j in range(0,len(return_pack[i][4])):
            com_alls[j].extend(return_pack[i][4][j])
        accus = np.sum([accus,return_pack[i][0]],axis=0)
        total_labels = np.sum([total_labels,return_pack[i][1]],axis=0)
        if bad_centroids is not None:
            wrongs += return_pack[i][2]
            bad_wrongs += return_pack[i][3]

    for i in range(0,total_classes):
        if total_labels[i]>0:
            for j in range(0,k):
                accus[j][i] = accus[j][i]/total_labels[i]
        else:
            for j in range(0,k):
                accus[j][i]=1.0
    #acc = np.mean(accus)
    acc = [np.mean(accus[j]) for j in range(k)]
    if bad_centroids is None:
        wrongs = 1.0
    return acc,bad_wrongs/wrongs,com_alls

def confusion_centroids(centroids,centroids_depth,w_dep,distance_threshold):
    distance_threshold = 20
    confused_centroids = []
    confusion = [[0.0 for x in range(0,len(centroids))] for y in range(0,len(centroids))]
    bad_centroids = [[0 for x in range(0,len(centroids[y]))] for y in range(len(centroids))]
    for i in range(0,len(centroids)):
        for j in range(0,len(centroids[i])):
            for k in range(i+1,len(centroids)):
                dists = np.subtract(centroids[i][j],centroids[k])
                dists = np.linalg.norm(dists,axis=1)
                dists_dep = np.subtract(centroids_depth[i][j],centroids_depth[k])
                dists_dep = np.linalg.norm(dists_dep,axis=1)
                dist = [1/((dists[x]+(w_dep*dists_dep[x]))/2) if (dists[x]+(w_dep*dists_dep[x]))/2<distance_threshold else 0.0 for x in range(len(centroids[k]))]
                confusion[i][k] += sum(dist)
                if sum(dist)>0.0:
                    bad_centroids[i][j] = 1
                    for index in range(0,len(bad_centroids[k])):
                        bad_centroids[k][index] += dist[index]
    return confusion,bad_centroids

def get_silhouette(pack):
    x_train = pack[0]
    x_train_depth = pack[1]
    centroids = pack[2]
    centroids_depth = pack[3]
    w_dep = pack[4]
    image_indices = pack[5]
    silhouttes = deepcopy(image_indices)

    for i in range(0,len(image_indices)):
        for j in range(0,len(image_indices[i])):
            dists = predict_multiple_class([x_train[image_indices[i][j]],x_train_depth[image_indices[i][j]],centroids,centroids_depth,0,'euclidean',w_dep,None])
            own_dist = dists[i][0]
            del dists[i]
            sorted_dists = sorted(dists)
            other_dist = sorted_dists[0][0]
            silhouttes[i][j] = (other_dist-own_dist)/max([other_dist,own_dist])
    return silhouttes

def get_silhouette_other(pack):
    x_train = pack[0]
    x_train_depth = pack[1]
    centroids = pack[2]
    centroids_depth = pack[3]
    w_dep = pack[4]
    image_indices = pack[5]
    silhouttes = deepcopy(image_indices)
    own_class = pack[6]

    other_classes = deepcopy(image_indices)
    for i in range(0,len(image_indices)):
        for j in range(0,len(image_indices[i])):
            own_dist = predict_multiple_class([x_train[image_indices[i][j]],x_train_depth[image_indices[i][j]],[centroids[own_class][i]],
            [centroids_depth[own_class][i]],0,'euclidean',w_dep,None])
            own_dist = own_dist[0][0]
            dists = []
            for k in range(0,len(centroids)):
                if k!=own_class:
                    temp = predict_multiple_class([x_train[image_indices[i][j]],x_train_depth[image_indices[i][j]],centroids[k],
                    centroids_depth[k],k,'euclidean',w_dep,None])
                    dists.extend(temp)
            sorted_dists = sorted(dists)
            other_dist = sorted_dists[0][0]
            other_classes[i][j] = sorted_dists[0][1]
            silhouttes[i][j] = (other_dist-own_dist)/max([other_dist,own_dist])
    return [silhouttes,other_classes]

def finding_dists_hidden(pack):
    x_train = pack[0]
    x_train_depth = pack[1]
    cent = pack[2]
    cent_depth = pack[3]

    x_train_new = []
    for i in range(0,len(x_train)):
        x_train_new.append(np.array(temp))
    return x_train_new

def finding_dists(pack):
    x_train = pack[0]
    x_train_depth = pack[1]
    centroids = pack[2]
    centroids_depth = pack[3]

    how_many = round(len(x_train)/24)
    pack = []
    now=0
    while now<len(x_train):
        if now+how_many>=len(x_train):
            pack.append([x_train[now:len(x_train)],x_train_depth[now:len(x_train)],centroids,centroids_depth])
        else:
            pack.append([x_train[now:how_many+now],x_train_depth[now:how_many+now],centroids,centroids_depth])
        now+=how_many

    my_pool = Pool(25)
    return_pack = my_pool.map(finding_dists_hidden,pack)
    my_pool.close()
    x_train_new = []
    for i in range(0,len(return_pack)):
        x_train_new.extend(return_pack[i])
    return np.array(x_train_new)
