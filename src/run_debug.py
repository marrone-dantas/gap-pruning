from train import TrainModel
import torch.optim as optim
import torch.nn as nn
from gap_pruning import GapPruning
import os
import torch
from torchinfo import summary
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    actual_path = os.getcwd()
    print('Runing on:',actual_path)

    batch_size = 2

    models_arr   = ['alexnet']
    dataset_arr  = ['cifar10']
    classes_arr  = [10]
    sample_train = [35000]
    sample_val   = [2000]
    pruning_rate = 50
    epochs = 10
    skip_initial_training =  True
    finetune_rate = 0.2

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

            if (not skip_initial_training):

                #Train the model
                train_handler.train_model(epochs=epochs,sufix=models_arr[setup_idx]+'_'+dataset_arr[setup_idx], weight_path=actual_path+'/checkpoints')

                #Saving the log
                train_handler.save_hist(actual_path+'/backlog/'+models_arr[setup_idx]+'_'+dataset_arr[setup_idx]+'_output.csv')

                #Evaluate the model
                train_handler.evaluate_model(train_handler.arr_dataset[2] ,generate_log=True, path_log=actual_path+'/backlog', prefix='eval_'+models_arr[setup_idx]+'_'+dataset_arr[setup_idx]+'_')
            else:

                #Load best model
                train_handler.load_weights(actual_path+'/checkpoints/loss_weights_'+models_arr[setup_idx]+'_'+dataset_arr[setup_idx])

            #Generating summary
            trained_model = train_handler.model
            trained_model = trained_model.to('cuda')
            #This input is 224x224 by the transformations, maybe we need to chage later
            summary(trained_model, input_size=(1, 3, 224, 224), verbose=1)
    
            print("Creating GapPruning object...")
            # Initialize the GapPruning class with the model and dataset
            gap_pruning = GapPruning(model=trained_model, dataset=train_handler.arr_dataset[1], device='cuda')
            
            print("Registering hooks, and computing stds...")
            # Process the dataset to capture activation maps and compute standard deviations
            gap_pruning.process_dataset() 
            gap_pruning.compute_std_devs()

            print("Generating pruning proposal...")
            # Generate a pruning proposal based on the current pruning rate
            gap_pruning.generate_pruning_proposal(pruning_percentage=pruning_rate)

            print("Pruning the backbone...")
            # Apply pruning based on the generated proposal
            gap_pruning.prune()
            
            #Saving pruned backbone
            print("Saving pruned backbone...")
            torch.save(gap_pruning.model.state_dict(), actual_path+'/checkpoints/'+models_arr[setup_idx]+'_'+dataset_arr[setup_idx]+'_'+str(pruning_rate)+'.pth')
            
            #Removing hooks
            print("Removing hooks...")
            gap_pruning.remove_hooks()

            #Finetune model
            train_handler.model = gap_pruning.model
            train_handler.set_dataset(id_dataset=dataset_arr[setup_idx], batch_size=batch_size, num_train=int(sample_train[setup_idx]*finetune_rate), num_val=int(sample_val[setup_idx]*finetune_rate), download=True)
            train_handler.train_model(epochs=epochs,sufix='pruned_'+str(pruning_rate)+'_'+models_arr[setup_idx]+'_'+dataset_arr[setup_idx], weight_path=actual_path+'/checkpoints')
            train_handler.save_hist(actual_path+'/backlog/pruned_'+str(pruning_rate)+'_'+models_arr[setup_idx]+'_'+dataset_arr[setup_idx]+'_output.csv')
            train_handler.evaluate_model(train_handler.arr_dataset[2] ,generate_log=True, path_log=actual_path+'/backlog', prefix='pruned_'+str(pruning_rate)+'_eval_'+models_arr[setup_idx]+'_'+dataset_arr[setup_idx]+'_')

            #TODO
            #Add the conversion to Tensorflow Lite
            #Add the quantization model


    