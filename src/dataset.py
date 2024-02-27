import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import sampler
import torchvision.transforms as T
import torchvision.datasets as dset
from torch.utils.data import DataLoader

import torchvision.transforms as T
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

""" from datasets.tiny import TinyImagenet
from datasets.tin import TinyImageNetDataset """


#Get chunker of the sampler
class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start
    
    def __iter__(self):
        return iter(range(self.start, self.start+self.num_samples))
    
    def __len__(self):
        return self.num_samples

#Define basic data augmentation
def get_data_aug():
    transform_augment = T.Compose([
        T.Resize(256),  # Resize the smaller edge to 256 pixels
        T.CenterCrop(224),  # Crop the center 224x224 pixels
        T.RandomHorizontalFlip(),
    ])
    transform_normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    return transform_augment, transform_normalize


import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.datasets as dset
import torchvision.transforms as T

def get_loader(dataset_type, batch_size=1024, num_train=45000, num_val=5000, download=False):
    transform_augment, transform_normalize = get_data_aug()

    # Assuming get_data_aug() is defined elsewhere and returns appropriate transformations
    transform = T.Compose([transform_augment, transform_normalize])

    if dataset_type == 'cifar10':
        dataset = dset.CIFAR10('../datasets', train=True, download=download, transform=transform)
    elif dataset_type == 'cifar100':
        dataset = dset.CIFAR100('../datasets', train=True, download=download, transform=transform)
    elif dataset_type == 'flowers102':
        dataset = dset.Flowers102('../datasets', split='train', download=download, transform=transform)
    elif dataset_type == 'food101':
        dataset = dset.Food101('../datasets', split='train', download=download, transform=transform)
    else:
        raise ValueError("Unsupported dataset type")

    # Create indices for splitting the dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = num_train
    validation_indices = indices[split:split+num_val]

    # Creating data samplers
    train_sampler = SubsetRandomSampler(indices[:num_train])
    validation_sampler = SubsetRandomSampler(validation_indices)

    # Creating data loaders
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valloader = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler)

    # For the test set
    if dataset_type in ['cifar10', 'cifar100']:
        test_transform = T.Compose([transform_normalize, transform_augment])  # Assuming no augmentation for test set
        test_dataset = dset.__dict__[dataset_type.upper()]('../datasets', train=False, download=download, transform=test_transform)
    else:
        # Adjust accordingly for other datasets like Flowers102 or Food101
        test_dataset = dataset  # Placeholder, adjust based on actual availability

    testloader = DataLoader(test_dataset, batch_size=batch_size)

    classes = None
    if dataset_type in ['cifar10', 'cifar100']:
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, valloader, testloader, classes




def get_processed_loader(root='../datasets/ProcessedMiniImagenet', batch_size=32, num_train=45000, num_val=5000):

    transform_augment, transform_normalize = get_data_aug()

    # Training set with augmentation
    train_dataset = datasets.ImageFolder(os.path.join(root, 'Train'), transform=transforms.Compose([transform_augment, transform_normalize]))
    
    # Since we don't have a predefined train/val split, use SubsetRandomSampler for validation
    train_sampler = SubsetRandomSampler(range(num_train))
    val_sampler = SubsetRandomSampler(range(num_train, num_train + num_val))

    trainloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valloader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

    # Test set (no augmentation, just normalization)
    test_dataset = datasets.ImageFolder(os.path.join(root, 'Test'), transform=transform_normalize)
    testloader = DataLoader(test_dataset, batch_size=batch_size)

    return trainloader, valloader, testloader, None

def get_dataset(id_dataset='cifar10', batch_size=1024, num_train=45000, num_val=5000, download=False):
    
    
    
    if (id_dataset=='cifar10'):

        print('Loading CIFAR 10')
        return get_loader(dataset_type=id_dataset, batch_size=batch_size, num_train=num_train, num_val=num_val, download=download)

    if (id_dataset=='cifar100'):

        print('Loading CIFAR 100')
        return get_loader(dataset_type=id_dataset,batch_size=batch_size, num_train=num_train, num_val=num_val, download=download)
    
    if (id_dataset=='miniimagenet'):

        print('Loading MINIIMAGENET')
        current_path = os.getcwd()
        return get_processed_loader(root=current_path+'/datasets/ProcessedMiniImagenet', batch_size=batch_size, num_train=num_train, num_val=num_val)
    
    if (id_dataset=='food101'):
        
        print('Loading FOOD101')
        return get_loader(dataset_type=id_dataset,batch_size=batch_size, num_train=num_train, num_val=num_val, download=download)
    
    