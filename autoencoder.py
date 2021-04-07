import torch
import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler as sch
from torch.utils.data import DataLoader, TensorDataset
import utils
from models import ConvAutoencoder
import torchvision.transforms.functional as tff



def read_image_data(game):
    all_images = []

    pl_list = utils.get_players_only_file(game)
    lines = pl_list.readlines()
    for l in lines:
        image = utils.read_and_process(utils.prefix+l.strip())
        all_images.append(image) 


    return all_images


def load_data(games):
    for i, game in enumerate(games):
        x = read_image_data(game)
       
        if i == 0:
            all_x = x
        else:
            all_x = all_x+x



    all_x = np.array(all_x,dtype=np.uint8)
    tensor_x = torch.stack([tff.to_tensor(i) for i in all_x] )
    print(str(tensor_x.size()))
    dataset = TensorDataset( tensor_x)        
    loader = DataLoader(dataset, batch_size=100, shuffle=False)
    
    return loader 
    
def train_model(model_name, train_loader, val):
    # initialize the NN
    model = ConvAutoencoder()
    model.cuda()


    # specify loss function
    criterion = nn.MSELoss()

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    # number of epochs to train the model
    n_epochs = 300
    min_val_loss = 1000
    n_epochs_stop = 3
    epochs_no_improve = 0
    val_loss = 0
    scheduler = sch.StepLR(optimizer, step_size=100, gamma=0.1)
    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0
                   
        for data in train_loader:
            imgs_in = data[0]
            #$print(images)
            imgs_in = imgs_in.type(torch.FloatTensor)
            imgs_in = imgs_in.cuda()


            optimizer.zero_grad()
            outputs = model(imgs_in)
            loss = criterion(outputs, imgs_in)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*imgs_in.size(0)
        

        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        
        #get validation loss
        val_loss = 0.0
        for data in val_loader:
            imgs_in = data[0]
            #$print(images)
            imgs_in = imgs_in.type(torch.FloatTensor)
            imgs_in = imgs_in.cuda()

            optimizer.zero_grad()
            outputs = model(imgs_in)
            loss = criterion(outputs, imgs_in)

            val_loss += loss.item()*imgs_in.size(0)

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


    torch.save(model.state_dict(),utils.trained_models_dir+model_name)

PER_GAME_TRAIN_DATASET_SIZE = 6000
if __name__== "__main__":    
    save_as = 'autoencoder_test.pth'
    train_loader = load_data(utils.train_games)
    val_loader = load_data(utils.val_games) 
    train_model(save_as, train_loader,val_loader)
    







    
    
    
    