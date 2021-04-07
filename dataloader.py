#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:58:01 2020

@author: maria
"""

import torch
import numpy as np
import numpy.matlib as npm
import cv2 as cv
import torchvision.transforms.functional as tff
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from models import ContrastNN
import utils

###### Constants ###############
PER_GAME_TRAIN_DATASET_SIZE = 8000
CONF_THRESHOLD = 0.9

######### Flags #############
debug = False
augment = True
max_images  = PER_GAME_TRAIN_DATASET_SIZE*2 if augment else PER_GAME_TRAIN_DATASET_SIZE


def read_image_data(games, do_augment=True):
    all_images = []
    for i, game in enumerate(games):
        pl_list = utils.get_players_only_file(game)
        lines = pl_list.readlines()
        images = []
        for l in lines:
            image = utils.read_and_process(utils.prefix+l.strip())
            images.append(image) 
            if do_augment:   
                images.append(utils.get_flipped_image(image))         
        if len(images) > max_images:
            images = images[:max_images]
        all_images.append(images)
    return all_images

# get soft clustering weights
def soft_clustering_weights(data, cluster_centres):
    # Fuzziness parameter m>=1. Where m=1 => hard segmentation
    m = 2
    
    Nclusters = cluster_centres.shape[0]
    Ndp = data.shape[0]

    # Get distances from the cluster centres for each data point and each cluster
    EuclidDist = np.zeros((Ndp, Nclusters))
    for i in range(Nclusters):
        EuclidDist[:,i] = np.sum((data-npm.repmat(cluster_centres[i], Ndp, 1))**2,axis=1)
    
    
    # Denominator of the weight from wikipedia:
    invWeight = EuclidDist**(2/(m-1))*npm.repmat(np.sum((1./EuclidDist)**(2/(m-1)),axis=1).reshape(-1,1),1,Nclusters)
    Weight = 1./invWeight
    
    return Weight

# return data for game and high-confidence cluster subsets
def load_data_helper(game, model, game_imgs, use_hist, threshold = CONF_THRESHOLD):       
    if use_hist:
        features = utils.get_hist_features(game_imgs)
    else:
        features = utils.get_features(game_imgs, model)
    
    #do clustering
    kmeans = KMeans(n_clusters=2).fit(features)
    
    #calculate soft clustering responsibilites  
    probs = soft_clustering_weights(np.asarray(features), kmeans.cluster_centers_)
    
    #get a subset with high probability for each cluster
    indx1 = np.where(probs[:,0] > CONF_THRESHOLD)
    indx2 = np.where(probs[:,1] > CONF_THRESHOLD) 
    size_cl1 = len(indx1[0])
    size_cl2 = len(indx2[0])
    
    print('high confidence ratio: ' + str((size_cl1 + size_cl2)/max_images))

    return indx1, indx2


def to_grayscale(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_new = np.zeros_like(img)
    img_new[:,:,0] = gray
    img_new[:,:,1] = gray
    img_new[:,:,2] = gray
    return img_new

#load data in triplets: anchor, same team, different team
def load_data_triplet (games, model, images, use_hist, threshold = CONF_THRESHOLD):
    if len(images)==0:
            images = read_image_data(games, do_augment=augment)
    X1 = []
    X2 = []
    X3 = []
    for i, game in enumerate(games):
        game_imgs = images[i]
        indx1, indx2 = load_data_helper(game, model, game_imgs, use_hist, threshold=threshold)
        size_cl1 = len(indx1[0])
        size_cl2 = len(indx2[0])  
        
         # make positive pairs for each cluster
        num_pos_cl1 = size_cl1 // 2
        num_pos_cl2 = size_cl2 // 2
        pos_pairs1 = np.random.choice(indx1[0], (num_pos_cl1, 2), replace=False)
        pos_pairs2 = np.random.choice(indx2[0], (num_pos_cl2, 2), replace=False)
        neg_third1 = np.random.choice(indx2[0], (num_pos_cl1, 1), replace=True)
        neg_third2 = np.random.choice(indx1[0], (num_pos_cl2, 1), replace=True)
        pos_pairs = np.concatenate((pos_pairs1, pos_pairs2), axis=0)
        negatives = np.concatenate((neg_third1, neg_third2), axis=0)
        for k, p in enumerate(pos_pairs):
            flip = 0#random.randint(0, 1)
            if flip == 0:
                img1 = game_imgs[p[0]]
                img2 = game_imgs[p[1]]
                img3 = game_imgs[negatives[k][0]]
            else:               
                # randomly turn some of the triplets images into grayscale
                img1 = to_grayscale(game_imgs[p[0]])
                img2 = to_grayscale(game_imgs[p[1]])
                img3 = to_grayscale(game_imgs[negatives[k][0]])
            X1.append(img1)
            X2.append(img2)
            X3.append(img3)
            if debug and k < 5:
                print('triplets:')
                img1 = np.array(img1,dtype=np.uint8)
                img2 = np.array(img2,dtype=np.uint8)
                img3 = np.array(img3,dtype=np.uint8)
            
                plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))      
                plt.show(block=False)
                plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))      
                plt.show(block=False)
                plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))      
                plt.show(block=False)
                print('---------------------------')
    X1 = np.array(X1,dtype=np.uint8)
    X2 = np.array(X2,dtype=np.uint8)
    X3 = np.array(X3,dtype=np.uint8)
    tensor_x1 = torch.stack([tff.to_tensor(i) for i in X1] )
    tensor_x2 = torch.stack([tff.to_tensor(i) for i in X2] )
    tensor_x3 = torch.stack([tff.to_tensor(i) for i in X3] )
    print(str(tensor_x1.size()))
    dataset = TensorDataset( tensor_x1, tensor_x2, tensor_x3 )        
    loader = DataLoader(dataset, batch_size=100, shuffle=True)
    
    return loader, images          

#testing code
if __name__== "__main__":
    #games = ["game5"]
    games = utils.train_games
    model_path = utils.trained_models_dir + 'embedding.pth'
    #loader = load_data (games, [], [], True)
    model = ContrastNN()        
    if utils.isCuda:
        model.load_state_dict(torch.load(model_path))
        model.to(torch.device('cuda'))
    else:
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()
    loader = load_data_triplet (games, model, [], False)