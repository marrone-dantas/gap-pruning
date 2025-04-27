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
from torchvision.models.detection import ssd, ssd300_vgg16
from PIL import Image
from torchinfo import summary
import os
import copy
from gap_pruning import GapPruning
from torch import jit
from dataset import get_dataset

def get_ssd_custom(backbone):
    
    #Modifiing model
    modified_vgg16 = copy.deepcopy(backbone.features)

    # Define SSD model using the custom backbone
    num_classes = 2  # Number of classes in COCO dataset
    anchor_generator = torchvision.models.detection.anchor_utils.DefaultBoxGenerator([[1,2,3,4,5]])
    size = (300, 300)  # Input size of the SSD model
    ssd_model = ssd.SSD(backbone=modified_vgg16, num_classes=num_classes, anchor_generator=anchor_generator, size=size)

    return ssd_model

def get_jist_model(model):
    example_input  = torch.rand(1, 3, 300, 300)
    traced_model   = torch.jit.trace(model, example_input)
    return traced_model

class SSDWrapper(nn.Module):
    def __init__(self, model):
        super(SSDWrapper, self).__init__()
        self.model = model
    
    def forward(self, images):
        detections = self.model(images)
        # Assuming a single image input for simplicity; adjust if using batches
        boxes = detections[0]['boxes']
        labels = detections[0]['labels']
        scores = detections[0]['scores']
        # Format the output as a tuple of tensors
        return boxes, labels, scores


# Define a list of pruning rates to evaluate
list_pruning_rates = [1,10,20,80,90,]
flg_load = False

# Iterate over each pruning rate with a progress bar
for rate_ in tqdm(list_pruning_rates, desc="Pruning Rates Progress"):
    
    # Load the pretrained VGG16 model
    print("Loading model..." )
    model = torch.load('/media/marrone/M2/Projects/gap_compress/checkpoints/acc_full_model_vgg_miniimagenet.pt')
    model = model.to('cuda')
    summary_str = summary(model, input_size=(1, 3, 224, 224), verbose=0).__repr__()
    summary_str = summary_str.splitlines()
    summary_str = "\n".join(summary_str[-11:])
    print(summary_str)
    
    # Define the transformations for CIFAR-10 dataset images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to fit VGG16 input size
        transforms.ToTensor()           # Convert images to PyTorch tensors
    ])
    
    print("Loading dataset...")
    # Load Mini-ImageNet dataset
    mini_data = get_dataset(id_dataset='miniimagenet', batch_size=8, num_train=100, num_val=500, download=False)
    
    print("Creating GapPruning object...")
    # Initialize the GapPruning class with the model and dataset
    gap_pruning = GapPruning(model=model, dataset=mini_data[0], device='cuda')
    
    print("Registering hooks, and computing stds...")
    # Process the dataset to capture activation maps and compute standard deviations
    gap_pruning.process_dataset() 
    gap_pruning.compute_std_devs()

    print("Generating pruning proposal...")
    # Generate a pruning proposal based on the current pruning rate
    gap_pruning.generate_pruning_proposal(pruning_percentage=rate_)

    print("Pruning the backbone...")
    # Apply pruning based on the generated proposal
    gap_pruning.prune()
    
    #Saving pruned backbone
    print("Saving pruned backbone...")
    torch.save(gap_pruning.model.state_dict(), 'checkpoints/ssd_model_pruned_weights_'+str(rate_)+'.pth')
    
    #Removing hooks
    print("Removing hooks...")
    gap_pruning.remove_hooks()
    
    #Creating SSD model with pruned backbone    
    print("Saving model with pruning rate: ", rate_)
    ssd_model = get_ssd_custom(gap_pruning.model)
    
    # Create an example input (here assuming SSD300 expects 300x300 images)
    #Generating JIST -GPU
    device = torch.device('cuda')
    print("Exporting JIST model on device: ", device)
    wrapped_model = SSDWrapper(ssd_model).to(device).eval()
    example_input = torch.rand(1, 3, 300, 300).to(device)
    traced_model   = torch.jit.trace(wrapped_model, example_input)
    jit.save(traced_model, 'checkpoints/ssd_model_jist_device_'+str(device)+'_pruned_'+str(rate_)+'.pth')
    
        # Create an example input (here assuming SSD300 expects 300x300 images)
    #Generating JIST - CPU
    device = torch.device('cpu')
    print("Exporting JIST model on device: ", device)
    wrapped_model = SSDWrapper(ssd_model).to(device).eval()
    example_input = torch.rand(1, 3, 300, 300).to(device)
    traced_model   = torch.jit.trace(wrapped_model, example_input)
    jit.save(traced_model, 'checkpoints/ssd_model_jist_device_'+str(device)+'_pruned_'+str(rate_)+'.pth')
    
    #Exporting quantizide model - Quantization TODO
    '''
    print("Exporting quantizide model...")
    quant_back = gap_pruning.quantize_model()
    ssd_model_quant = get_ssd_custom(quant_back)
    traced_model   = torch.jit.trace(ssd_model_quant, example_input)
    traced_model.save('checkpoints/ssd_model_jist_quantizide_'+str(rate_)+'.pth')
    '''
    