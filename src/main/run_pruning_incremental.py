from train import TrainModel
import torch.optim as optim
import torch.nn as nn
from gap_pruning_interactive import GapPruningInteractive as GapPruning
import os
import torch
from torchinfo import summary
import warnings
from telegram_handler import *
from rich.console import Console
import copy

warnings.filterwarnings("ignore")

console = Console()

if __name__ == '__main__':

    actual_path = os.getcwd()
    console.print(f"[bold blue]Running on:[/bold blue] [white]{actual_path}[/white]")

    batch_size = 2

    models_arr   = [
    "resnet50.ra_in1k"
    ]
    dataset_arr  = ['cifar10','cifar100', 'flowers102', 'food101']
    classes_arr  = [10,100, 102, 101]
    pruning_rates = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    epochs = 250
    base_model_paths = '/media/marronedantas/HD4TB/Projects/gap-pruning/checkpoints/base_trained_models'
    base_pruned_model_paths = '/media/marronedantas/HD4TB/Projects/gap-pruning/checkpoints/base_prunned_models'

    for model_name in models_arr:
        for dataset_name, num_classes in zip(dataset_arr, classes_arr):
            
            # Setting dummy trainer with base model
            dummy_train = TrainModel()
            dummy_train.set_model(id_model=model_name, num_classes=num_classes)
            dummy_train.set_dataset(id_dataset=dataset_name, batch_size=batch_size, download=True)

            console.rule(
                f"[bold magenta] Model:[/bold magenta] [white]{model_name}[/white]  •  "
                f"[bold magenta]Dataset:[/bold magenta] [white]{dataset_name}[/white]"
            )
            
            # Gettin the weights or the model
            model_path = f'{base_model_paths}/{model_name}/loss_full_model_{model_name}_{dataset_name}.pt'
            
            try:
                
                dummy_train.load_weights(model_path)
                console.rule(f"[bold magenta] Model Weights Loaded:[/bold magenta] [white]{model_name}[/white]  •  ")
                
            except:
                
                dummy_train.model = torch.load(model_path, weights_only=False)
                console.rule(f"[bold magenta] Full Model Loaded:[/bold magenta] [white]{model_name}[/white]  •  ")
            
            # Evaluate the original model
            optimizer = optim.SGD(dummy_train.model.parameters(),lr=0.001, momentum=0.9)
            dummy_train.set_optimizer(optimizer)
            dummy_train.set_criterion(nn.CrossEntropyLoss())

            scheduler = optim.lr_scheduler.MultiStepLR(dummy_train.optimizer, milestones=[75, 150], gamma=0.5)
            dummy_train.set_scheduler(scheduler)


            dummy_train.evaluate_model(
                            dummy_train.arr_dataset[1],
                            generate_log=True,
                            path_log=f"{base_pruned_model_paths}",
                            prefix=f"benchmark_{model_name}_{dataset_name}_"
                        )
            
            console.print("[magenta]▸ Starting GapPruning…[/magenta]")
            
            for pruning_rate in pruning_rates:

                # Reseting base model
                base_model = copy.deepcopy(dummy_train.model)
                gap_pruning = GapPruning(model=base_model, dataset=dummy_train.arr_dataset[0], device='cuda')

                for _ in range(pruning_rate):

                    gap_pruning.model = gap_pruning.model.to('cuda')
                    
                    # Compute the activations
                    gap_pruning.compute_stats_on_the_fly(flg_load=True)
                    
                    #Pruning process
                    pruned_report = gap_pruning.prune_online(prune_rate=1)
                    
                    # Finetune pruned model
                    dummy_train.model = copy.deepcopy(gap_pruning.model)
                    
                    # Setting optimizer and loss function
                    optimizer = optim.SGD(dummy_train.model.parameters(),lr=0.001, momentum=0.9)
                    dummy_train.set_optimizer(optimizer)
                    dummy_train.set_criterion(nn.CrossEntropyLoss())

                    scheduler = optim.lr_scheduler.MultiStepLR(dummy_train.optimizer, milestones=[75, 150], gamma=0.5)
                    dummy_train.set_scheduler(scheduler)
                
                    # Training the pruned model
                    dummy_train.train_model(
                                epochs=int(epochs/200),
                                )
                    
                    input()

                    