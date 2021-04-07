import torch
import numpy as np
import torchvision.transforms.functional as tf
import torch.nn as nn
import torch.optim.lr_scheduler as sch
from torch.utils.data import DataLoader, TensorDataset
import os
import utils
import time

import sys
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from models import ConvClassifier

import copy
from os.path import isfile, join
from os import listdir

##################################################################
# Referee Classifier;
# to train: python referee_classifier.py
# to test: python referee_classifier.py --test
# to save players (non-referees for each fold): 
#             python referee_classifier.py --save
# to save precision-recall for a range of thresholds: 
#             python referee_classifier.py --test --curve
#
##################################################################


isCuda = utils.isCuda
MAX_TESTING_IMGS = 860 #max number to use for evaluation - ensures same number for
                      #of images is used from each game
THRESHOLD = 0.5

if not isCuda:
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

################################# Data handling ######################### 
def get_files(path):
    all_files = []
    for f in listdir(path):
        if isfile(join(path,f)) and f[0]!='.':
            all_files.append(path+'/'+f)
    return all_files

#read max 8000 images for game
def read_all_imgs(game):
    X = []
    img_folder = utils.data_dir + game+ utils.images_sub_dir     
    img_file_paths = get_files(img_folder)
    
    if (len(img_file_paths) > 8000):
        img_file_paths = img_file_paths[:8000]
    
    for f in img_file_paths:
        image = utils.read_and_process(f)
        X.append(image)

    return X, img_file_paths


# read ground truth annotated images and labels for game
def read_annotated_imgs(game, augment=False):    
    #read ground truth 
    f_gt = utils.get_gt_file (game)
    gt = f_gt.readlines()
    
    y = []
    X = []
    count = 0
    for line in gt:
        g_tmp = line.split(',')
        name = str(g_tmp[0])  

        if (g_tmp[1].strip() == '2'):
            label = 1 
            count+=1
        else:
            label = 0

        name = utils.data_dir+ game+utils.images_sub_dir + name
        image = utils.read_and_process(name)
        image = np.array(image)
        X.append(image)
        y.append(label)
 
        #augment with horizontally flipped data (if this is for training)
        if augment:            
            X.append(utils.get_flipped_image(image))    
            y.append(label)
    print(count)
    return X, y


#takes list of games for training
def load_data_training(games, val_games=[]):
    X = []
    val = []
    Y = []
    y_val = []
    for game in games:
        x, y = read_annotated_imgs(game, augment=True)
        X.extend(x)
        Y.extend(y)
    
    if len(val_games) > 0:
        for game in val_games:
            x, y = read_annotated_imgs(game)
            val.extend(x)
            y_val.extend(y)    
    
    print(len(Y))
    X = np.array(X, dtype=np.uint8)

    tensor_x = torch.stack([tf.to_tensor(i) for i in X] )
    tensor_y = torch.tensor(Y)
    dataset = TensorDataset( tensor_x, tensor_y )
    train_loader = DataLoader(dataset, batch_size=100,  shuffle=True)
    
    val = np.array(val, dtype=np.uint8)
    tensor_val = torch.stack([tf.to_tensor(i) for i in val] )
    tensor_y_val = torch.tensor(y_val)
    dataset_val = TensorDataset( tensor_val, tensor_y_val )
    val_loader = DataLoader(dataset_val, batch_size=100,  shuffle=True)
    
    return train_loader, val_loader

#takes list of games for training
def load_data_testing(games):
    X = []
    Y = []
    for game in games:
        x, y = read_annotated_imgs(game)
        X.extend(x)
        Y.extend(y)

    X = np.array(X[:MAX_TESTING_IMGS], dtype=np.uint8)
    tensor_x = torch.stack([tf.to_tensor(i) for i in X] )
    tensor_y = torch.tensor(Y[:MAX_TESTING_IMGS])
    
    return tensor_x, tensor_y

####################################### Training code #######################    

def train_model(model_name, train_loader, val_loader):
    # initialize the NN
    model = ConvClassifier()
    if isCuda:
        model.cuda()

    # loss function
    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # number of epochs to train the model
    n_epochs = 1000
    min_val_loss = 1000
    n_epochs_stop = 5
    epochs_no_improve = 0
    val_loss = 0
    scheduler = sch.StepLR(optimizer, step_size=10, gamma=0.1)
    
    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0
        i = 0
                   
        for data in train_loader:
            i = i + 1
            images, y = data

            images = images.type(torch.FloatTensor)
            if isCuda:
                images = images.cuda()

            optimizer.zero_grad()
            outputs = model(images)

            y=y.reshape(-1,1)
            y = y.type(torch.FloatTensor)
            if isCuda:
                y = y.cuda()
            
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()


        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        
        #get validation loss
        for data in val_loader:
            images, y = data
            #$print(images)
            images = images.type(torch.FloatTensor)
            y=y.reshape(-1,1)
            y = y.type(torch.FloatTensor)
            if isCuda:
                images = images.cuda()
                y = y.cuda()

            optimizer.zero_grad()
            outputs = model(images)

            out = model(images)
            loss = criterion(out, y)
            val_loss += loss.item()


        # print avg training statistics 
        val_loss = val_loss/len(val_loader)
        print('Epoch: {} \tValidation Loss: {:.6f}'.format(epoch, val_loss))
        
        scheduler.step()
        
        if val_loss < min_val_loss:
             epochs_no_improve = 0
             min_val_loss = val_loss  
        else:
            epochs_no_improve += 1
        # Check early stopping condition
        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!' )
            break             

    torch.save(model.state_dict(), model_name)

##################################### Testing and stats ###################

# calculate precision recall for a specific threshold
def get_precision_recall_for_th (Y, results, th):
    binary_results = [0 if a < th else 1 for a in results]
    prec = precision_score(Y, binary_results)
    recall = recall_score(Y, binary_results)
    return prec, recall


def precision_recall_curve(Y, results):
    precision_list = []
    recall_list = []
    thresholds = np.arange(0.01, 0.99, 0.01).tolist()
    for th in thresholds:
        p, r = get_precision_recall_for_th (Y, results, th)
        precision_list.append(p)
        recall_list.append(r)
    return precision_list, recall_list

def test_model(model_name, X, Y, prec_recall_curve = False):
    # initialize the NN
    model = get_model('',model_name=model_name )
    length = len(Y)
    results = []

    for i, image in enumerate(X):   
        image = image.type(torch.FloatTensor)
        image =  image.unsqueeze(0)
        if isCuda:
            image = image.cuda()          
        if prec_recall_curve:
            out = model.predict(image, threshold = -1)
            out = out.detach().cpu().numpy()
        else:
            out = model.predict(image, threshold = THRESHOLD)
            out = out.detach().cpu().numpy()
        results.append(out[0])
    
    p, r = [], []
    acc = 0    
    if prec_recall_curve:    
        p, r = precision_recall_curve(Y, results)    
    else:
        Y = Y.data.numpy()
        acc = (results == Y).sum()/length   
        print("Accuracy:" + str(acc))
        
        tn, fp, fn, tp = confusion_matrix(Y, results).ravel()
        prec  = tp / (tp + fp)
        recall = tp / (tp + fn)
        print ("Precision:"+str(prec))
        print ("Recall:"+str(recall))
        
    return acc, p, r

def test_classification(model, game, prec_recall_curve = False):
    # initialize the NN
    model = get_model('', model_name=model)
    X, Y = load_data_testing([game])
    
    length = len(Y)
    results = []

    for i, image in enumerate(X):   
        image = image.type(torch.FloatTensor)
        image =  image.unsqueeze(0)
        if isCuda:
            image = image.cuda()          
        if prec_recall_curve:
            out = model.predict(image, threshold = -1)
            out = out.detach().cpu().numpy()
        else:
            out = model.predict(image, threshold = THRESHOLD)
            out = out.detach().cpu().numpy()
        results.append(out[0])
    
    p, r = [], []
    acc = 0    
    if prec_recall_curve:    
        p, r = precision_recall_curve(Y, results)    
    else:
        Y = Y.data.numpy()
        acc = (results == Y).sum()/length   
        print("Accuracy:" + str(acc))
        
    return acc, p, r

def get_model (suffix, model_name = ''):
    if model_name == '':
        model_name = utils.trained_models_dir + 'referee_classifier_segments_'+suffix+'.pth' 
    model = ConvClassifier()
    
    if isCuda:   
        model.load_state_dict(torch.load(model_name))
        model.cuda()
    else: 
        model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
        
    model.eval()
    return model

#given a game return only non-referees
def get_players_only_list (game, suffix, model_name='', save = False, file_name = '', threshold = THRESHOLD):
    new_X = []
    if save:
        if file_name == '':
            file_name = utils.data_dir+game+ utils.players_only_file
        results_file = open(file_name, 'w')
    X, file_names = read_all_imgs(game)
    if model_name == '':
        model = get_model(suffix)
    else:
        model = get_model('', model_name=model_name)
      
    for i, image in enumerate(X):   
        original = copy.deepcopy(image)
        tensor_x = tf.to_tensor(image)
        image = tensor_x.type(torch.FloatTensor)
        image =  image.unsqueeze(0)
        if isCuda:
            image = image.cuda()            
        out = model.predict(image, threshold = threshold)
        out = out.detach().cpu().numpy()
        if (out[0]==0):
            new_X.append(original)
            if save:
                results_file.write(file_names[i]+ "\n")    
            
    return new_X

################################## Main ########################################

if __name__== "__main__":    
    isTrain = True
    getPrecisionRecallCurve = False
    save = False
    all_precisions = []
    all_recalls = []
    games_acc = []

                
    if len(sys.argv) > 1:
        if (sys.argv[1].strip() == '--test'):
            isTrain = False
            if len(sys.argv) > 2 and sys.argv[2].strip() == '--curve':
                getPrecisionRecallCurve = True
        elif sys.argv[1].strip() == '--save':
            save = True
            isTrain = False
      
  
    model_name = utils.trained_models_dir + 'referee_classifier_segments.pth'        
    if isTrain:          
        loader, val_loader = load_data_training(utils.train_games, val_games = utils.val_games) 
        train_model(model_name, loader, val_loader)
    elif save:
        all_games = utils.train_games + utils.test_games + utils.val_games
        for game in all_games:
            print('---------------Saving results: '+ game + ' ---------------------------')

            get_players_only_list (game, '', model_name=model_name, save=True)
    else:
        for i, game in enumerate(utils.test_games):
            print("for "+game+":")
            X, Y = load_data_testing([game])
            a, curve_p, curve_r = test_model(model_name, X, Y, prec_recall_curve = getPrecisionRecallCurve)
            if getPrecisionRecallCurve:
                all_precisions.append(curve_p)
                all_recalls.append(curve_r)
            else:
                games_acc.append(a)

    if (not isTrain) and getPrecisionRecallCurve:      
        mean_prec_curve=np.mean(all_precisions, axis=0)
        mean_recall_curve=np.mean(all_recalls, axis=0)
        std_prec = np.std(all_precisions, axis=0)
        std_recall = np.std(all_recalls, axis=0)
        
        #dump this into a stats file
        timestamp = time.time()
        f = open('stats/ref_stats_'+str(timestamp)+'.txt', 'w')
        f.write(str(mean_prec_curve)+'\n')
        f.write(str(mean_recall_curve)+'\n')
        f.write(str(std_prec)+'\n')
        f.write(str(std_recall))
        f.close()
    elif (not isTrain) and not save:
        mean_acc = np.mean(games_acc)
        print('Mean accuracy is ' + str(mean_acc))
