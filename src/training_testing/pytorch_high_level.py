import os
import cv2
import numpy as np 
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import sleep

GPU = True
if torch.cuda.is_available():
    # torch.cuda.device_count -> for counting number of gpus. most laptops will have just 1
    device = torch.device("cuda:0")# currently only supporting 1 gpu
    GPU = True
    print('running on GPU')
else:
    device = torch.device("cpu")
    GPU = False
    print("running on CPU")

def fwd_pass(net,X,Y,optimizer,loss_function,train=False):
    if train:
        net.zero_grad()
    output = net(X)
    if(output.shape != Y.shape):
        print("output shape does not match target shape!")
        print("input shape:",X.shape)
        print("output shape:",output.shape)
        print("target shape:",Y.shape)
        exit()
    loss = loss_function(output,Y)
    output = None
    del output
    if train:
        loss.backward()
        optimizer.step()
    return loss

def fit(net,X,Y,train_log,optimizer,loss_function,validation_set,BATCH_SIZE,EPOCHS,model_name):
    early_stopping = EarlyStopping()

    val_size = int(validation_set*len(X))
    data_size = len(X)
    train_size = data_size - val_size

    for epochs in range(EPOCHS):
        print('epoch ', epochs)
        sleep(0.1)
        #insample data
        train_average_loss = 0
        val_average_loss = 0
        train_counter = 0
        val_counter = 0
        optimizer = optim.Adam(net.parameters(),lr = 0.001)
        loss_function = nn.MSELoss()
        for i in tqdm(range(0,train_size, BATCH_SIZE ) ):
            batch_X = (X[i:i+BATCH_SIZE]).to(device)
            batch_Y = (Y[i:i+BATCH_SIZE]).to(device)
            train_loss = fwd_pass(net,batch_X,batch_Y,optimizer,loss_function,train=True)
            batch_X = None
            del batch_X
            batch_Y = None
            del batch_Y
            if i%100==0:
                train_average_loss += float(train_loss.cpu())
                train_counter += 1
            train_loss = None
            del train_loss
        #outsample data
        del optimizer,loss_function
        torch.cuda.empty_cache()

        optimizer = optim.Adam(net.parameters(),lr = 0.001)
        loss_function = nn.MSELoss()
        for i in tqdm(range(train_size,data_size,BATCH_SIZE)):
            batch_X = (X[i:i+BATCH_SIZE]).to(device)
            batch_Y = (Y[i:i+BATCH_SIZE]).to(device)
            val_loss = fwd_pass(net,batch_X,batch_Y,optimizer,loss_function,train=False)
            batch_X = None
            del batch_X
            batch_Y = None
            del batch_Y
            if i%10==0:
                val_average_loss += float(val_loss.cpu())
                val_counter += 1
            val_loss = None
            del val_loss
            # print('val loss: ',float(val_loss))
        torch.cuda.empty_cache()
        if(train_counter==0):
            train_counter = 1
        if(val_counter ==0):
            val_counter = 1
        train_log.append([train_average_loss/train_counter,val_average_loss/val_counter]) # just store the last values for now

        optimizer = None
        loss_function = None

        del optimizer, loss_function
        torch.cuda.empty_cache()

        state = {'net': net}
        torch.save(state, model_name, _use_new_zipfile_serialization=False)

        print('train loss = ', train_average_loss/train_counter)
        print('val loss = ', val_average_loss/val_counter)
        early_stopping(val_average_loss/val_counter)
        if early_stopping.early_stop:
            break

    return train_log       


def fit_dataloader(net,DL_train, DL_val, train_log,EPOCHS,model_name):
    early_stopping = EarlyStopping()

    for epochs in range(EPOCHS):
        print('epoch ', epochs)

        #insample data
        train_average_loss = 0
        val_average_loss = 0
        train_counter = 0
        val_counter = 0

        optimizer = optim.Adam(net.parameters(),lr = 0.001)
        loss_function = nn.MSELoss()

        for (i, batch) in enumerate(DL_train):
            # print("\nBatch = " + str(i))
            batch_X = batch['predictors']  # [3,7]
            batch_Y = batch['political']  # [3]

            train_loss = fwd_pass(net,batch_X,batch_Y,optimizer,loss_function,train=True)
            del batch_X
            del batch_Y

            if i%100==0:
                train_average_loss += float(train_loss.cpu())
                train_counter += 1

            del train_loss

        #outsample data
        del optimizer, loss_function
        torch.cuda.empty_cache()

        optimizer = optim.Adam(net.parameters(), lr=0.001)
        loss_function = nn.MSELoss()

        for (i, batch) in enumerate(DL_val):
            # print("\nBatch = " + str(i))
            batch_X = batch['predictors']  # [3,7]
            batch_Y = batch['political']  # [3]

            val_loss = fwd_pass(net,batch_X,batch_Y,optimizer,loss_function,train=False)
            del batch_X
            del batch_Y

            if i%10==0:
                val_average_loss += float(val_loss.cpu())
                val_counter += 1

            del val_loss
            # print('val loss: ',float(val_loss))

        torch.cuda.empty_cache()
        if(train_counter==0):
            train_counter = 1
        if(val_counter ==0):
            val_counter = 1
        train_log.append([train_average_loss/train_counter,val_average_loss/val_counter]) # just store the last values for now

        del optimizer, loss_function
        torch.cuda.empty_cache()

        state = {'net': net}
        torch.save(state, model_name, _use_new_zipfile_serialization=False)

        print('train loss = ', train_average_loss/train_counter)
        print('val loss = ', val_average_loss/val_counter)
        early_stopping(val_average_loss/val_counter)
        if early_stopping.early_stop:
            break

    return train_log


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print("INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
