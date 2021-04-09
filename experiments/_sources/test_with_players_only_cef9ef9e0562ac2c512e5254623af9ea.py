#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:24:19 2020

@author: maria
"""
import math
import numpy as np
import utils
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from joblib import dump, load
from sacred import Experiment
from sacred.observers import FileStorageObserver




ex = Experiment()
ex.observers.append(FileStorageObserver('experiments'))

@ex.config
def config():    
    
    #method = 'bag'  
    method = 'net' #embedding network
    #method = 'hist' 
    #method = 'ae'
    
    model_name = 'embedding'
    model_version = ''
    ae_model_path = utils.trained_models_dir+'autoencoder.pth'
    
    image_w = 64
    image_h = 128
    
    #config related to bag-of-colors approach
    K = 35
    h = 64
    pixel_number = 400000
    use_learned_bag = False # use vocabulary learned on other games
    
    frames_train_size = 512
    #number of frames to use for learning clustering
    #[1,2,4,8,16,32,64,128, 256, 512]
    
    # step is used for k-fold testing indicates offset for picking training frames 
    # for example step = 10 means pick training frames for each test run from 
    # run 1: from frame #1, run 2: from frame #11, run 3: from frame #21, etc
    #
    # to run once from the first frame set step to 0
    step = 0

    gt_frames_number = 30  #number of gt frames to use for evaluation
    
    
    store_clusters = False # store learned team clusters
    
    save_bbox_labels = False # update bbox and labels file, that we use for 
                            # heatmap generation
    

    all_games = utils.train_games + utils.val_games + utils.test_games
                   


@ex.capture
def read_sorted_train_image_data(game, offset, image_w, image_h, frames_train_size):
    image_size=(image_w, image_h)
    
    return utils.read_sorted_train_image_data(game, offset, image_size, frames_train_size, augment=False)

@ex.capture
def get_bag_features_from_existing (train_images, test_images, game, K):
    gmm = load(utils.trained_models_dir +'train_set_colors.joblib')
    X_train = utils.get_hist_from_gmm(gmm,train_images, K=K)
    X_test = utils.get_hist_from_gmm(gmm, test_images, K=K)
    return X_train, X_test     
  
@ex.capture
def get_bag_features(train_images, test_images, game, K, h, pixel_number, store_clusters, use_learned_bag):
    #encode using colors learned on other game
    if use_learned_bag:
        return get_bag_features_from_existing (train_images, test_images, game)
    
    # learn bag of colors on the current game
    flat = []
    for i, c in enumerate(train_images):
        c = c[:h,:,:]
        r = c[:,:,0].flatten()
        b = c[:,:,1].flatten()
        g = c[:,:,2].flatten()
        temp = np.dstack((r,b,g))
        non_black_pixels_mask = np.any(temp != [0, 0, 0], axis=-1)
        flat.extend(temp[non_black_pixels_mask].tolist())
        if len(flat) > pixel_number:
            break
    if len(flat) > pixel_number: 
        flat = flat[:pixel_number]      
    gmm = GaussianMixture(n_components=K).fit(flat)  
    
    if store_clusters:
        dump(gmm, utils.trained_models_dir + game+'_colors.joblib') 
    
    
    X_train = utils.get_hist_from_gmm(gmm,train_images, K=K)
    X_test = utils.get_hist_from_gmm(gmm, test_images, K=K)
    return X_train, X_test     

@ex.capture
def read_test_image_data(game, offset, image_w, image_h, frames_train_size, gt_frames_number):
    image_size=(image_w, image_h)
       
    return utils.read_test_image_data(game, offset, image_size, frames_train_size, gt_frames_number)

def save_labels(f, names, labels):
    for i, n in enumerate(names):
        f.write(n+','+str(labels[i])+'\n')

def write_bbox_labels(video, names, labels, suffix):
    file = open(utils.data_dir +video+ '/bbox_data.txt', 'r')
    f_new = open(utils.data_dir +video+ '/players_label_bbox_data_'+suffix+'.txt', 'w')
    lines = file.readlines()
    for j, fr in enumerate(lines):
        if (j+1)%10 != 0:
            continue
        fr = fr.strip()
        fr = fr[:-1]
        boxes = fr.split(';')
        out = ''
        for b in boxes:
            tmp = b.split(',')
            name = tmp[4]
            name = utils.data_dir +video +utils.images_sub_dir +name           
            try:
                indx = names.index(name)
            except:
                continue
            
            b = b + ','+str(labels[indx])+';'
            out = out + b
        f_new.write(out+'\n')
    f_new.close()
    file.close()

#run for single game - determine cluster centers based on <frames_train_size> number
#of frames using features extracted with method specified by <method>;
#predict labels for the next <gt_frames_number> frames
@ex.capture
def run_for_game (game, model, offset, method, store_clusters):
    train_images = read_sorted_train_image_data(game, offset)   
    test_images, gt_clusters, test_img_names = read_test_image_data(game, offset)
       
    if (method=='hist'):
        train_features = utils.get_hist_features(train_images)
        test_features = utils.get_hist_features(test_images)
    elif (method=='bag'):
        train_features, test_features = get_bag_features(train_images, test_images, game)   
    else:
        train_features = utils.get_features(train_images, model)
        test_features = utils.get_features(test_images, model)


    kmeans = KMeans(n_clusters=2).fit(train_features)
    if store_clusters:
        dump(kmeans, utils.trained_models_dir+ game+'_colors_kmeans_clusters.joblib')             
    labels = kmeans.predict(test_features)

    return test_img_names, labels, gt_clusters


@ex.automain
def main(method, model_name, model_version, save_bbox_labels, frames_train_size, step, ae_model_path):   
    useKfold = False
    if step > 0:
        # used for k-fold testing accuracy calculation
        useKfold = True
        games_stats = []
        for k in range(len(utils.test_games)):
            games_stats.append([])
        
        
    acc = []
    print("Running with method " + method)

    #Evaluate method for all test games
    for j, game in enumerate(utils.test_games):
        model_path = utils.trained_models_dir+model_name+model_version+'.pth'  
        if (method == 'net'):
            model = utils.load_model_embed(model_path)
        elif (method=='ae'):
            model = utils.load_model_embed(ae_model_path, isAE=True)
        else:
            model =[]

        offset = 0
        while offset <= 512-frames_train_size:
            print(" running for game " + game + " starting from frame " + str(offset+1))                    
                   
            names, labels, gt_clusters = run_for_game(game, model, offset)  
               
            result_clusters= [[] for y in range(2)]
            for m in range(len(labels)):
                result_clusters[labels[m]].append(names[m])  
            
            two_way_acc = utils.get_stats(gt_clusters, result_clusters, 2)
            
            if useKfold:
                games_stats[j].append(two_way_acc)
            else:
                acc.append(two_way_acc)
                
            if save_bbox_labels:
                labels_file_path = 'results/'+ game + '_labels_'+ method+'.txt'
                f_results = open(labels_file_path, 'w') 
                save_labels(f_results, names, labels)   
                write_bbox_labels(game, names, labels, method)
                f_results.close()
                ex.add_artifact(labels_file_path)
            if step == 0:
                offset = 512
            else:
                offset = offset + step
    if (useKfold):
        acc = np.mean(games_stats, axis=1)
    print("mean acc: " + str(np.mean(acc)))
    print("mean error: " + str(1-np.mean(acc)))
    print("std acc: " + str(np.std(acc)))
    print("standard error: " + str(np.std(acc)/math.sqrt(len(utils.test_games))))