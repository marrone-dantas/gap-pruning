from train import TrainModel

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import os
import warnings
warnings.filterwarnings("ignore")

#Getting the curent path 
current_path = os.getcwd()
print('Runing on:',current_path)

#Setting model and dataset
train_handler = TrainModel()
train_handler.set_model(id_model='vgg',num_classes=100)
train_handler.set_dataset(id_dataset='miniimagenet', batch_size=48, num_train=100, num_val=4000, download=True)

#Setting optimizer and criterion  
optimizer = optim.SGD(train_handler.model.parameters(), lr=0.001, momentum=0.9)      
criterion = nn.CrossEntropyLoss()           

#Set optimizer and criterion
train_handler.set_optimizer(optimizer)
train_handler.set_criterion(criterion)

#Setting learnign scheduler
scheduler = optim.lr_scheduler.MultiStepLR(train_handler.optimizer, milestones=[75, 150], gamma=0.5)
train_handler.set_scheduler(scheduler)

#Train the model
train_handler.train_model(epochs=1,sufix='vgg_miniimagenet', weight_path=current_path+'/checkpoints')

#Saving the log
train_handler.save_hist(current_path+'/backlog/vgg_miniimagenet_output.csv')

#Load best model
train_handler.load_weights(current_path+'/checkpoints/loss_weights_vgg_miniimagenet')

#Evaluate the model
train_handler.evaluate_model(train_handler.arr_dataset[2],path_log=current_path+'/backlog/eval_vgg_miniimagenet')