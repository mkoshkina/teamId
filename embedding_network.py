import torch
import torch.nn as nn
import torch.optim.lr_scheduler as sch
import utils
from dataloader import load_data_triplet
from models import ContrastNN

################ Training code ########################
def train_model(model, train_loader, val, n_epochs, lr):
    Loss = nn.TripletMarginLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    min_val_loss = 1000
    n_epochs_stop = 3
    epochs_no_improve = 0
    val_loss = 0
    scheduler = sch.StepLR(optimizer, step_size=10, gamma=0.5)
    done = False
    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0
                   
        for data in train_loader:
            imgs1, imgs2, imgs3 = data

            imgs1 = imgs1.type(torch.FloatTensor)
            imgs2 = imgs2.type(torch.FloatTensor)
            imgs3 = imgs3.type(torch.FloatTensor)
            if isCuda:
                imgs1 = imgs1.cuda()
                imgs2 = imgs2.cuda()
                imgs3 = imgs3.cuda()

            optimizer.zero_grad()
            out1 = model.encode(imgs1)
            out2 = model.encode(imgs2)  
            out3 = model.encode(imgs3)                      
            loss = Loss(out1, out2, out3)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*imgs1.size(0)
        

        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        
        #get validation loss
        val_loss = 0.0
        for data in val_loader:
            imgs1, imgs2, imgs3 = data
            imgs1 = imgs1.type(torch.FloatTensor)
            imgs2 = imgs2.type(torch.FloatTensor)
            imgs3 = imgs3.type(torch.FloatTensor)
            if isCuda:
                imgs1 = imgs1.cuda()
                imgs2 = imgs2.cuda()
                imgs3 = imgs3.cuda()

            optimizer.zero_grad()
            out1 = model.encode(imgs1)
            out2 = model.encode(imgs2)
            out3 = model.encode(imgs3) 
            
            loss = Loss(out1, out2, out3)

            val_loss += loss.item()*imgs1.size(0)

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
        if (epochs_no_improve == n_epochs_stop) or train_loss == 0.0 :
            print('Early stopping!' )
            done = True
            break             


    return model, done

#### Training Constants ##############
isCuda = utils.isCuda
train_images = []
val_images = []
epochs = 10
max_epochs = 500
gamma = 0.5
debug = True


if __name__== "__main__":
    model = ContrastNN()   
    
    if isCuda:
        model.to(torch.device('cuda'))
            
    model.eval()
    lr = 0.0001
    
    if debug:
        print("Histogram accuracy for training set:" + str(utils.evaluate_clustering (utils.train_games, '')))
        print("Histogram accuracy for validation set:" + str(utils.evaluate_clustering (utils.val_games, '')))
    
    #pre-train on histogram clustering triplets for 30 epochs (or until convergence)
    train_loader, train_images = load_data_triplet(utils.train_games, model, train_images, True)
    val_loader, val_images = load_data_triplet(utils.val_games, model, val_images,True) 
    model, _ = train_model(model, train_loader, val_loader, epochs*3, lr)
    
    if debug:
        print("After 1st stage accuracy for training set:" + str(utils.evaluate_clustering (utils.train_games, model)))
        print("After 1st stage accuracy for validation set:" + str(utils.evaluate_clustering (utils.val_games, model)))
    
    success = False
    epoch_counter = epochs
    # Keep training until val loss no longer imrpoves
    # Regenerate data triplets every 10 epochs
    stage = 1
    while not success and epoch_counter <= max_epochs:
        train_loader, train_images = load_data_triplet(utils.train_games, model, train_images, False, threshold=0.9)
        val_loader, val_images = load_data_triplet(utils.val_games, model, val_images, False, threshold=0.9) 
        model, success = train_model(model, train_loader, val_loader, epochs, lr)
        epoch_counter = epoch_counter + epochs
        lr = lr * gamma
        stage += 1
        if debug:
            print("After stage" + str(stage) +" accuracy for training set:" + str(utils.evaluate_clustering (utils.train_games, model)))
            print("After stage" + str(stage) +" accuracy for validation set:" + str(utils.evaluate_clustering (utils.val_games, model)))
    
    model_name = utils.trained_models_dir+'embedding.pth'
    torch.save(model.state_dict(),model_name) 
        







    
    
    
    