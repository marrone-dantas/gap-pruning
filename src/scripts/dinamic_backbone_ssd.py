import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as transF
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch_pruning as tp
import torchvision.models as models
import torchvision
from torchvision.models.detection import ssd
from PIL import Image
import os
import copy
from src.gap_pruning import GapPruning

def get_ssd_custom(backbone, path_model = 'ssd_model_weights.pth'):
    
    #Modifiing model
    modified_vgg16 = copy.deepcopy(backbone.features)

    # Define SSD model using the custom backbone
    num_classes = 2  # Number of classes in COCO dataset
    anchor_generator = torchvision.models.detection.anchor_utils.DefaultBoxGenerator([[1,2,3,4,5]])
    size = (300, 300)  # Input size of the SSD model
    ssd_model = ssd.SSD(backbone=modified_vgg16, num_classes=num_classes, anchor_generator=anchor_generator, size=size)

    # Set the model to evaluation mode
    ssd_model.eval()

    #Save model
    torch.save(ssd_model,path_model)

    return None


# Define a list of pruning rates to evaluate
list_pruning_rates = [1, 10, 20, 30, 95, 99]
flg_load = True

# Iterate over each pruning rate with a progress bar
for rate_ in tqdm(list_pruning_rates, desc="Pruning Rates Progress"):
    
    # Load the pretrained VGG16 model
    model = models.vgg16(pretrained=True)
    
    # Define the transformations for CIFAR-10 dataset images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to fit VGG16 input size
        transforms.ToTensor()           # Convert images to PyTorch tensors
    ])
    
    # Load CIFAR-10 dataset
    cifar10_data = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    
    # Initialize the GapPruning class with the model and dataset
    gap_pruning = GapPruning(model=model, dataset=cifar10_data, device='cuda')
    
    # Process the dataset to capture activation maps and compute standard deviations
    if (flg_load):
        gap_pruning.compute_std_devs(flg_load=flg_load)
    else:
        gap_pruning.process_dataset()  # Assuming process_dataset accepts a samples_size argument
        gap_pruning.compute_std_devs(flg_load=flg_load)

    # Generate a pruning proposal based on the current pruning rate
    gap_pruning.generate_pruning_proposal(pruning_percentage=rate_)
    
    # Apply pruning based on the generated proposal
    gap_pruning.prune()
    
    # Save the pruned model's state dict for later use or evaluation
    # Assuming get_ssd_custom is a function that saves the model (the name suggests SSD, but seems to be used for VGG16 here)
    pruned_ssd = get_ssd_custom(gap_pruning.model, path_model='checkpoints/ssd_model_weights_'+str(rate_)+'.pth')