#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:42:21 2020

@author: maria
"""
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from models import ContrastNN, ConvAutoencoder
import torch
import torchvision.transforms.functional as tf
import copy
from imgaug import augmenters as iaa


################# Constants ##############################
isCuda = torch.cuda.is_available()

train_games = ["video0", "video1", "video2", "video5", "game1","game2", "game3", "game5", "game6"]
val_games = ["video8", "game7"]
test_games = ["video3", "video10", "game8", "game9"]

data_dir = 'data/'
images_sub_dir = '/masked_imgs/'
gt_file_name = '/gt.txt'
players_only_file = '/players_only.txt'
prefix=''

trained_models_dir = '../trained_models/'


IMAGE_SIZE = (64,128)
######################################################

def get_players_only_file (game):
    return open(data_dir+game+players_only_file, 'r')

def get_gt_file (game):
    return open(data_dir+game+gt_file_name, 'r')

def load_model_embed(model_path, isAE = False):  
    #load atoencoder/feature extraction model
    if isAE:
        model = ConvAutoencoder()
    else:
        model = ContrastNN()        
    if isCuda:
        model.load_state_dict(torch.load(model_path))
        model.to(torch.device('cuda'))
    else:
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()

    return model 

def get_hist_features(images):
    X = []
    for image in images:
        img_hist = []
        non_black_pixels_mask = np.any(image != [0, 0, 0], axis=-1)
        for k in range(image.shape[2]):
            temp = image[:,:,k]
            new = temp[non_black_pixels_mask].flatten()
            #new = temp[temp!=0].flatten()
            hist, _ = np.histogram(new,8)
            hist = hist/(1.0*np.sum(hist))
            
            img_hist = img_hist + hist.tolist()

        X.append(img_hist)
    return X

def get_hist_from_gmm(gmm,images, K=30, useColocation=False):
    X = []
    for c in images:
        c = c[:64,:,:]
        r = c[:,:,0].flatten()
        b = c[:,:,1].flatten()
        g = c[:,:,2].flatten()
        temp = np.dstack((r,b,g))
        non_black_pixels_mask = np.any(temp != [0, 0, 0], axis=-1)
        temp = temp[non_black_pixels_mask].tolist()
        
        hist = np.zeros(K, dtype=np.float)
        if len(temp)!=0:
            labels = gmm.predict(temp)        
            for i in (range(K)):
                hist[i]=np.sum(labels==i)
            
        histSum = np.sum(hist)
        hasNans = np.isnan(histSum)
                
        if hasNans or histSum == 0:
            feature=np.zeros(K, dtype=np.float)
        else:
            feature = hist/(1.0*histSum) 
        X.append(feature)
    return X

def get_features(images, model):
    X = []
    for image in images:
        image = tf.to_tensor(image) 
        image =  image.unsqueeze(0)
        if isCuda:
            image = image.cuda()
        out = model.encode(image)
        out = out.detach().cpu().numpy()
        X.append(out[0])
    return X

def get_key(video_name, frame):
    return video_name+'_'+str(frame)+'_'

def frame_index(key, files):
    result = []
    for i in range(len(files)):
        if key in files[i]:
            result.append(i)
    return result

def get_frame_id_from_name(name, video_name):
    tmp = name.split('_')
    return int(tmp[1])
 
def read_train_image_data(game, offset, image_size, per_game_train_size):
    pl_list = get_players_only_file(game)
    lines = pl_list.readlines()
    images = []
    for l in lines:
        image = read_and_process(l.strip(), image_size, top_portion=True)           
        if (image is None):
            continue
        images.append(image) 
         
    if len(images) > per_game_train_size:
        images = images[:per_game_train_size]
    return images


def read_test_image_data(game, offset, image_size, frames_train_size, gt_frames_number):
    gt = get_gt_file(game)
    lines = gt.readlines()

    images = []
    names = []
    gt_clusters= [[] for x in range(2)]
       
    for line in lines:
        g_tmp = line.split(',')
        gt_k = int(g_tmp[1]) - 1
        if gt_k == 1:
            continue
        elif gt_k == 2:
            gt_k = 1
        name=str(g_tmp[0]) 
        frame_id = get_frame_id_from_name(name, game)
        
        #only read portion of ground truth if we want to limit how we evaluate 
        #accuracy
        if gt_frames_number > 0:
            if frame_id < offset + frames_train_size:
                continue
            if frames_train_size > 0 and frame_id > (gt_frames_number*10 + offset + frames_train_size):
                break
        
        name=data_dir+game+images_sub_dir+name

        image = read_and_process (name, image_size)        
        if (image is None):
            continue
        #name=game+'/crops/'+name
        gt_clusters[gt_k].append(name)
        names.append(name)
     
        images.append(image) 
    
    return images, gt_clusters, names   

def read_and_process (image_path, image_size = IMAGE_SIZE, top_portion=False):
    image = cv.imread(image_path) 

    temp = image
    if top_portion:
        temp=image[:64,:,:]
    #print(image_path)
    non_black_pixels_mask = np.any(temp != [0, 0, 0], axis=-1)
    extracted = temp[non_black_pixels_mask].tolist()
    if (image is None) or len(extracted)==0:
        return None
    
    image = cv.resize(image, image_size)
        
    # normalize image to change brightness
    image = cv.normalize(image,None,0.0,255.0,norm_type=cv.NORM_MINMAX,dtype=cv.CV_32F)
    image = image.astype(np.uint8)
    
    return image

def get_flipped_image (image):
     original = copy.deepcopy(image)       
     rotate = iaa.Fliplr(1.0) # horizontally flip
     augmented = rotate.augment_image(original)
     return augmented


def read_sorted_train_image_data(game, offset, image_size, frames_train_size, augment=False):
    pl_list = get_players_only_file(game)
    lines = pl_list.readlines()
    names = []
    for l in lines:
        names.append(prefix+l.strip())
    names, _ = sort_by_frame(game, names, frames_train_size, offset)   
    
    n = len(names)
    images = []

    for i in range(n):
        image = read_and_process(names[i])
        images.append(image)
        if (augment):           
            images.append(get_flipped_image(image))   

    return  images

def sort_by_frame(game, files, frames_train_size, offset):
    #sort by frame:
    i = 0
    new_files = []
    #frame_ids = np.zeros(len(files))
    frame_ids = []
    frame = 1
    frame_count = 0
    if not offset == 0:
        frame = offset + 1
    while len(new_files) < len(files):
        if frame_count >= frames_train_size:
            break;
        key = get_key(game, frame)
        indx = frame_index(key, files)
        if len(indx) == 0:
            frame = frame + 1
            continue
        for j in indx:
            new_files.append(files[j])
            #frame_ids[i] = frame
            frame_ids.append(frame)
            i = i+1
        frame += 1
        frame_count += 1
    return new_files, frame_ids


def get_stats(result_clusters, gt_clusters, cluster_n) :
    accuracy_vector = []
    options=np.matrix('0,1;1,0')
    for o in range(len(options)):
        correct_n=0
        total=0 
        opt = options[o,:]
        for i in range(cluster_n): 
              total_in_cluster = len(gt_clusters[i])
              correct_in_cluster=0
                        
              #cluster index varies per run
              r_i=opt[0,i]   
              for img_name in gt_clusters[i]:
                  try:
                      indx=result_clusters[r_i].index(img_name)
                      correct_in_cluster=correct_in_cluster+1
                  except:
                      continue
              total = total + total_in_cluster
              correct_n = correct_n + correct_in_cluster
        accuracy_vector.append(correct_n/total)  
    accuracy = np.max(accuracy_vector)
    print("accuracy: " + str(accuracy))   
        
    return accuracy

def evaluate_clustering_per_game (game, model):
    # read all gt, cluster and report accuracy
    images, gt_clusters, names = read_test_image_data(game, 0, IMAGE_SIZE, 0, 0)
    if model == '':
        features = get_hist_features(images)
    else:
        features = get_features(images, model)
    labels = KMeans(n_clusters=2).fit_predict(features)
    result_clusters= [[] for y in range(2)]
    for m in range(len(labels)):
        result_clusters[labels[m]].append(names[m])  
            
    return get_stats(gt_clusters, result_clusters, 2)

def evaluate_clustering (games, model):
    acc = []
    for game in games:
        acc.append(evaluate_clustering_per_game (game, model))
    return np.mean(acc), np.min(acc)