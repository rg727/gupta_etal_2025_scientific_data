# -*- coding: utf-8 -*-
"""
Created on Fri May  9 17:58:49 2025

@author: rg727
"""


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import time
from datetime import date
import os


#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LSTM Configuration and Hyperparameters   
num_input_feature = 34
num_output_feature = 1

num_epochs = 30
num_hidden_layer = 1
squence_length = 365
num_hidden_neuron = 256
batch_size = 64
learning_rate = 0.0005
dropout_rate = 0.4

if not os.path.exists('/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_5/model/'):
        os.makedirs('/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_5/model/')
        print("Created folder")
else:
        print("Folder already exists")

if not os.path.exists('/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_5/prediction/'):
        os.makedirs('/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_5/prediction/')
        print("Created folder")
else:
        print("Folder already exists")

if not os.path.exists('/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_5_neighbors/model/'):
        os.makedirs('/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_5_neighbors/model/')
        print("Created folder")
else:
        print("Folder already exists")

if not os.path.exists('/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_5_neighbors/prediction/'):
        os.makedirs('/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_5_neighbors/prediction/')
        print("Created folder")
else:
        print("Folder already exists")

if not os.path.exists('/scratch/bcqp/rg727/1950_1965_Training_Testing/Testing/out_5/OOS/'):
        os.makedirs('/scratch/bcqp/rg727/1950_1965_Training_Testing/Testing/out_5/OOS/')
        print("Created folder")
else:
        print("Folder already exists")

if not os.path.exists('/scratch/bcqp/rg727/1950_1965_Training_Testing/Testing/out_5_neighbors/OOS/'):
        os.makedirs('/scratch/bcqp/rg727/1950_1965_Training_Testing/Testing/out_5_neighbors/OOS/')
        print("Created folder")
else:
        print("Folder already exists")               


seed_all = [1000000]
#seed_all = [1000000,2000000,3000000,4000000,5000000,6000000,7000000,8000000,9000000,10000000]
for seed in seed_all:

    torch.manual_seed(seed)
    
    ########## Data Period for Training
    f_date = date(1950,1,1)
    l_date = date(1960,12,31)
    
    
    ########## Training set statistics for Standardization
    t_start = time.time()
    basin_list_train = np.loadtxt('/scratch/bcqp/rg727/1950_1965_Training_Testing/fold_5_calibration.txt',dtype=str)   
    n_day_train = (l_date-f_date).days+1
    n_day_tot = len(basin_list_train)*n_day_train
    
    n = 0
    feature_train = np.zeros(shape=[n_day_tot, num_input_feature],dtype=np.float32)
    for basin_id in basin_list_train:   
        xy = np.loadtxt('/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_5/'+basin_id+'.csv',delimiter=",", dtype=np.float32, skiprows=1)
        feature_train[n:n+n_day_train,:] = xy[:n_day_train,4:]
        n += n_day_train
       
    t_end = time.time()    
    elapsed_time = t_end - t_start  
    
    feature_train_mean = np.mean(feature_train,axis=0)
    feature_train_std = np.std(feature_train,axis=0)
    print(f'Elapse time for calculating training set statistics: {elapsed_time:.2f} sec')
    

    ########## LSTM input sequences for Training Basins
    basin_list_train = np.loadtxt('/scratch/bcqp/rg727/1950_1965_Training_Testing/fold_5_calibration.txt',dtype=str)   
    n_day = ((l_date-f_date).days+1)-squence_length+1
    n_day_tot = len(basin_list_train)*n_day
    
    
    y_seq_train = np.zeros(shape=[n_day_tot, num_output_feature],dtype=np.float32)
    n = 0
    n_day_arr = np.repeat(n_day,basin_list_train.shape[0])
    t_start = time.time()
    for k, basin_id in enumerate(basin_list_train):
        xy = np.loadtxt('/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_5/'+basin_id+'.csv',delimiter=",", dtype=np.float32, skiprows=1)
        y = xy[:n_day_train,[3]]  
        
        y_seq = y[squence_length-1:]
        y_seq_train[n:n+n_day,:] = y_seq
        
        n_day_arr[k] -= np.isnan(y_seq).sum()
        n += n_day
        
        t_end = time.time()    
        elapsed_time = t_end - t_start 
        print(f'Target sequences generated for Train Basin "{basin_id}", Elapsed time: {elapsed_time:.2f} sec')
    
    nan_boolean = np.isnan(y_seq_train)
    nan_ind = [i for i, x in enumerate(nan_boolean) if x]
    y_seq_train = np.delete(y_seq_train,nan_ind,0)
    
    
    x_seq_train = np.zeros(shape=[n_day_arr.sum(), squence_length, num_input_feature],dtype=np.float32)
    day_acc = 0
    t_start = time.time()
    for k, basin_id in enumerate(basin_list_train):
        xy = np.loadtxt('/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_5/'+basin_id+'.csv',delimiter=",", dtype=np.float32, skiprows=1)
        y = xy[:n_day_train,[3]]
        x = xy[:n_day_train,4:]    
        
        y_seq = y[squence_length-1:]
        x_seq = np.empty((n_day_arr[k],squence_length,x.shape[1]),dtype=np.float32)
        
        n = 0
        for i, j in enumerate(np.arange(squence_length,x.shape[0]+1)):
            if not np.isnan(y_seq[i]):
                x_seq[n,:,:] = np.divide(np.subtract(x[(j-squence_length):j,:],feature_train_mean),feature_train_std)
                #x_seq[n,:,:34] = np.divide(np.subtract(x[(j-squence_length):j,:34],feature_train_mean[:34]),feature_train_std[:34])
                #x_seq[n,:,44:] = x[(j-squence_length):j,44:]
                #x_seq[n,:,[34]] = np.repeat(np.divide(np.subtract(x[j-1,[34]],feature_train_mean[34]),feature_train_std[34]),squence_length)
                #x_seq[n,:,[35]] = np.repeat(np.divide(np.subtract(x[j-1,[35]],feature_train_mean[35]),feature_train_std[35]),squence_length)
                #x_seq[n,:,[36]] = np.repeat(np.divide(np.subtract(x[j-1,[36]],feature_train_mean[36]),feature_train_std[36]),squence_length)
                #x_seq[n,:,[37]] = np.repeat(np.divide(np.subtract(x[j-1,[37]],feature_train_mean[37]),feature_train_std[37]),squence_length)
                #x_seq[n,:,[38]] = np.repeat(np.divide(np.subtract(x[j-1,[38]],feature_train_mean[38]),feature_train_std[38]),squence_length)
                #x_seq[n,:,[39]] = np.repeat(x[j-1,[39]],squence_length)
                #x_seq[n,:,[40]] = np.repeat(x[j-1,[40]],squence_length)
                #x_seq[n,:,[41]] = np.repeat(x[j-1,[41]],squence_length)
                #x_seq[n,:,[42]] = np.repeat(x[j-1,[42]],squence_length)
                #x_seq[n,:,[43]] = np.repeat(x[j-1,[43]],squence_length)
                n += 1
         
        x_seq_train[day_acc:day_acc+n_day_arr[k],:,:] = x_seq
        day_acc += n_day_arr[k]
        
        t_end = time.time()    
        elapsed_time = t_end - t_start 
        print(f'Input sequences generated for Train Basin "{basin_id}", Elapsed time: {elapsed_time:.2f} sec')
    

    
    ########## Build LSTM model      
    class LSTM(nn.Module):
        def __init__(self, num_input_feature, num_hidden_neuron, num_hidden_layer, num_output_feature, dropout_rate):
            super(LSTM, self).__init__()
            self.num_hidden_layer = num_hidden_layer
            self.num_hidden_neuron = num_hidden_neuron
            self.lstm = nn.LSTM(num_input_feature,num_hidden_neuron,num_hidden_layer,batch_first=True)
            self.fc = nn.Linear(num_hidden_neuron, num_output_feature)
            self.dropout = nn.Dropout(dropout_rate)
            self.relu = nn.ReLU()
            
        def forward(self,x): # x: [batch_size, seq, num_input_feature]
            h0 = torch.zeros(self.num_hidden_layer, x.size(0), self.num_hidden_neuron).to(device)
            c0 = torch.zeros(self.num_hidden_layer, x.size(0), self.num_hidden_neuron).to(device)
      
            out,_ = self.lstm(x, (h0,c0)) # out: batch_size, squence_lengthgth, num_hidden_neuron
            out = out[:, -1, :] # out (batch_size, num_hidden_neuron) for the last hidden
            out = self.dropout(out)
            out = self.fc(out)
            out = self.relu(out)
            return out

    
    model = LSTM(num_input_feature, num_hidden_neuron, num_hidden_layer, num_output_feature, dropout_rate).to(device)  
    
    
    # Loss and Optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    
    # Dataset & DataLoader for Train and Validation
    class SeqDataset(Dataset):
        
        def __init__(self, train=True):
            if train:
                self.x = torch.from_numpy(x_seq_train)
                self.y = torch.from_numpy(y_seq_train)
                self.n_samples = x_seq_train.shape[0]
            else:
                self.x = torch.from_numpy(x_seq_train)
                self.y = torch.from_numpy(y_seq_train)
                self.n_samples = x_seq_train.shape[0]
            
        def __getitem__(self, index):
            return self.x[index], self.y[index]
        
        def __len__(self):
            return self.n_samples    
        
    
        
    train_dataset = SeqDataset(train=True) 
    train_input = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
    # To check train_input  
    # dataiter = iter(train_input)
    # data = next(dataiter)
    # features, labels = data
    # print(features, labels)
    
    
    
    
    ########## Training Loop
    total_samples = train_dataset.n_samples
    n_iterations = math.ceil(total_samples/batch_size)
    loss_epoch = np.zeros(num_epochs+1)
    loss_iter = np.zeros(num_epochs*n_iterations)
    loss_iter_single = np.zeros(n_iterations)
    
    model.eval()
    with torch.no_grad():
        for i,(inputs,output) in enumerate(train_input):
            batch = inputs.to(device)
            target = output.to(device)
            
            # forward
            pred = model(batch)
            loss = loss_fn(pred, target)
            loss_iter_single[i] = loss
        
        loss_epoch[0] = loss_iter_single.mean()
        
    
    
    n = 0
    t_start = time.time()
    for epoch in range(num_epochs):
     
        model.train()
        for i,(inputs,output) in enumerate(train_input):
            
            batch = inputs.to(device)
            target = output.to(device)
            
            # forward
            pred = model(batch)
            loss = loss_fn(pred, target)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_iter_single[i] = loss
            loss_iter[n] = loss
            n+=1
            
            if (i+1) % 100 == 0:
                t_end = time.time()
                elapsed_time = t_end - t_start 
                print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, loss = {loss.item():.4f}, time = {elapsed_time:.2f} sec')
                
        loss_epoch[epoch+1] = loss_iter_single.mean()
        # SAVE MODEL
        torch.save(model.state_dict(),'/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_5/model/lstm_nbr_ec_gl_train_seed_'+str(seed)+'_e'+str(epoch+1)+'.pth')
    
    
    # SAVE
    np.savetxt('/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_5/model/loss_train_epoch_seed_'+str(seed)+'_ec.csv', loss_epoch, delimiter=',',header='train',comments='')
    #np.savetxt('/scratch/bcqp/sungwookwi/lstm_nbr_gl/loss_train_iter_seed_'+str(seed)+'.csv', loss_iter, delimiter=',',header='train',comments='')
                    

model = LSTM(num_input_feature, num_hidden_neuron, num_hidden_layer, num_output_feature, dropout_rate).to(device)      
model.load_state_dict(torch.load('/scratch/bcqp/rg727/1950_1965_Training_Testing/Training/fold_5/model/lstm_nbr_ec_gl_train_seed_1000000_e30.pth'))
model.to(device)
    
  ########## LSTM input sequences for Validation Basin - ALL

#Data Period
f_date = date(1950,1,1)
l_date = date(1965,12,31)

basin_list_valid = np.loadtxt('/scratch/bcqp/rg727/1950_1965_Training_Testing/fold_5_validation.txt', dtype=str) 
n_day = ((l_date-f_date).days+1)-squence_length+1
n_day_tot = len(basin_list_valid)*n_day


y_seq_valid_all = np.zeros(shape=[n_day_tot, num_output_feature],dtype=np.float32)  
n = 0
n_day_arr = np.repeat(n_day,basin_list_valid.shape[0])
t_start = time.time()
for k, basin_id in enumerate(basin_list_valid):
    xy = np.loadtxt('/scratch/bcqp/rg727/1950_1965_Training_Testing/Testing/out_5/'+basin_id+'.csv',delimiter=",", dtype=np.float32, skiprows=1)
    y = xy[:,[3]]
            
    y_seq = y[squence_length-1:]
    y_seq_valid_all[n:n+n_day,:] = y_seq
    
    #n_day_arr[k] -= np.isnan(y_seq).sum()
    n += n_day
    
    t_end = time.time()    
    elapsed_time = t_end - t_start 
    print(f'ALL Target sequences generated for Validation Basin "{basin_id}", Elapsed time: {elapsed_time:.2f} sec')

# nan_boolean = np.isnan(y_seq_valid)
# nan_ind = [i for i, x in enumerate(nan_boolean) if x]
# y_seq_valid = np.delete(y_seq_valid,nan_ind,0)


 
    
x_seq_valid_all = np.zeros(shape=[n_day_arr.sum(), squence_length, num_input_feature],dtype=np.float32) # number of input features = 39
day_acc = 0
t_start = time.time()
for k, basin_id in enumerate(basin_list_valid):
    xy = np.loadtxt('/scratch/bcqp/rg727/1950_1965_Training_Testing/Testing/out_5/'+basin_id+'.csv',delimiter=",", dtype=np.float32, skiprows=1)
    y = xy[:,[3]]
    x = xy[:,4:]    
         
    y_seq = y[squence_length-1:]
    x_seq = np.empty((n_day_arr[k],squence_length,x.shape[1]),dtype=np.float32)
    
    n = 0
    for i, j in enumerate(np.arange(squence_length,x.shape[0]+1)):
        #if not np.isnan(y_seq[i]):
        x_seq[n,:,:] = np.divide(np.subtract(x[(j-squence_length):j,:],feature_train_mean),feature_train_std)    
        #x_seq[n,:,:34] = np.divide(np.subtract(x[(j-squence_length):j,:34],feature_train_mean[:34]),feature_train_std[:34])
        #x_seq[n,:,44:] = x[(j-squence_length):j,44:]
        #x_seq[n,:,[34]] = np.repeat(np.divide(np.subtract(x[j-1,[34]],feature_train_mean[34]),feature_train_std[34]),squence_length)
        #x_seq[n,:,[35]] = np.repeat(np.divide(np.subtract(x[j-1,[35]],feature_train_mean[35]),feature_train_std[35]),squence_length)
        #x_seq[n,:,[36]] = np.repeat(np.divide(np.subtract(x[j-1,[36]],feature_train_mean[36]),feature_train_std[36]),squence_length)
        #x_seq[n,:,[37]] = np.repeat(np.divide(np.subtract(x[j-1,[37]],feature_train_mean[37]),feature_train_std[37]),squence_length)
        #x_seq[n,:,[38]] = np.repeat(np.divide(np.subtract(x[j-1,[38]],feature_train_mean[38]),feature_train_std[38]),squence_length)
        #x_seq[n,:,[39]] = np.repeat(x[j-1,[39]],squence_length)
        #x_seq[n,:,[40]] = np.repeat(x[j-1,[40]],squence_length)
        #x_seq[n,:,[41]] = np.repeat(x[j-1,[41]],squence_length)
        #x_seq[n,:,[42]] = np.repeat(x[j-1,[42]],squence_length)
        #x_seq[n,:,[43]] = np.repeat(x[j-1,[43]],squence_length)
        n += 1
    
    x_seq_valid_all[day_acc:day_acc+n_day_arr[k],:,:] = x_seq
    day_acc += n_day_arr[k]
     
    
    t_end = time.time()    
    elapsed_time = t_end - t_start 
    print(f'ALL Input sequences generated for Validation Basin "{basin_id}", Elapsed time: {elapsed_time:.2f} sec')       
    
    
    
n = 0
nse_valid = np.zeros((len(basin_list_valid),1))
for k, basin_id in enumerate(basin_list_valid):
    input = torch.from_numpy(x_seq_valid_all[n:n+n_day,:,:])
    target = torch.from_numpy(y_seq_valid_all[n:n+n_day,:])
    
    input = input.to(device)
    target = target.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input)
    
        valid_predictions = outputs.detach().cpu().numpy()
        valid_targets = target.detach().cpu().numpy()
    
        pred_obs = np.concatenate((valid_predictions, valid_targets),axis=1)
        np.savetxt('/scratch/bcqp/rg727/1950_1965_Training_Testing/Testing/out_5/OOS/lstm_nbr_pred_'+basin_id+'.csv', pred_obs, delimiter=',',header='pred,obs',comments='')
        
        nan_boolean = np.isnan(valid_targets)
        nan_ind = [j for j, x in enumerate(nan_boolean) if x]
        target_valid = np.delete(valid_targets,nan_ind,0)
        output_valid = np.delete(valid_predictions,nan_ind,0)
        
        nse = 1 - np.sum(np.power(target_valid - output_valid,2))/np.sum(np.power(target_valid - target_valid.mean(),2))
        nse_valid[k] = nse   
    
        n += n_day    
    
    
    np.savetxt('/scratch/bcqp/rg727/1950_1965_Training_Testing/Testing/out_5/OOS/nse_validation_cv_check.csv', nse_valid, delimiter=',',header='test',comments='')
   