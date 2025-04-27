from train import TrainModel
import torch.optim as optim
import torch.nn as nn
import os
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    actual_path = os.getcwd()
    print('Runing on:',actual_path)

    batch_size = 2

    models_arr  = [
    "vgg16.tv_in1k",
    "resnet50.ra_in1k",
    "efficientnet_b0.ra_in1k",
    "mobilenetv2_100.ra_in1k",
    "convnext_base.fb_in1k",
    "densenet121.ra_in1k",
    "regnety_032.ra_in1k",
    "repvgg_a0.rvgg_in1k",
    "swin_tiny_patch4_window7_224.ms_in1k",
    "mixnet_s.ft_in1k"
    ]
    dataset_arr  = ['miniimagenet','food101','cifar10','cifar100']
    classes_arr  = [100,101,10,100]
    sample_train = [30000,60000, 45000,45000]
    sample_val   = [5000,5000, 5000,5000]
    epochs = 50

    for setup_idw in range(len(models_arr)):
        
        for setup_idx in range(len(models_arr)):
            
            #Setting model and dataset
            train_handler = TrainModel()
            train_handler.set_model(id_model=models_arr[setup_idw],num_classes=classes_arr[setup_idx])
            train_handler.set_dataset(id_dataset=dataset_arr[setup_idx], batch_size=batch_size, num_train=sample_train[setup_idx], num_val=sample_val[setup_idx], download=True)

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
            train_handler.train_model(epochs=epochs,sufix=models_arr[setup_idx]+'_'+dataset_arr[setup_idx], weight_path=actual_path+'/checkpoints')

            #Saving the log
            train_handler.save_hist(actual_path+'/backlog/'+models_arr[setup_idx]+'_'+dataset_arr[setup_idx]+'_output.csv')

            #Load best model
            train_handler.load_weights(actual_path+'/checkpoints/loss_weights_'+models_arr[setup_idx]+'_'+dataset_arr[setup_idx])

            #Evaluate the model
            train_handler.evaluate_model(train_handler.arr_dataset[2] ,generate_log=True, path_log=actual_path+'/backlog', prefix='eval_'+models_arr[setup_idx]+'_'+dataset_arr[setup_idx]+'_')