import os
import random
from collections import defaultdict
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch

def save_n_samples_per_class(dataset_name: str, root_dir: str, output_dir: str, n_samples: int, image_size=(224, 224)):
    """
    Downloads a torchvision dataset and saves N samples per class as PNG images.

    Parameters:
        dataset_name (str): Name of the torchvision dataset (e.g., 'CIFAR10', 'CIFAR100', 'FOOD101', 'FLOWERS102').
        root_dir (str): Directory where the dataset will be downloaded.
        output_dir (str): Directory where the images will be saved.
        n_samples (int): Number of samples to save per class.
        image_size (tuple): Size to resize the images to.
    """

    # Dictionary mapping dataset names to torchvision dataset classes
    available_datasets = {
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100,
        'FOOD101': datasets.Food101,
        'FLOWERS102': datasets.Flowers102,
    }

    if dataset_name not in available_datasets:
        raise ValueError(f"Dataset '{dataset_name}' is not supported. Available options: {list(available_datasets.keys())}")

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # Load dataset
    DatasetClass = available_datasets[dataset_name]

    if dataset_name in ['FOOD101', 'FLOWERS102']:
        dataset = DatasetClass(root=root_dir, split='train', download=True, transform=transform)
    else:
        dataset = DatasetClass(root=root_dir, train=True, download=True, transform=transform)

    # Organize samples per class
    class_counts = defaultdict(int)
    class_to_idx = dataset.class_to_idx if hasattr(dataset, 'class_to_idx') else {str(i): i for i in range(len(set([label for _, label in dataset])))}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    os.makedirs(output_dir, exist_ok=True)

    for i, (img, label) in enumerate(dataset):
        if isinstance(label, torch.Tensor):
            label = label.item()

        if class_counts[label] >= n_samples:
            continue

        class_name = idx_to_class[label] if label in idx_to_class else str(label)
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        img_path = os.path.join(class_dir, f"{class_name}_{class_counts[label]}.png")
        save_image(img, img_path)

        class_counts[label] += 1

        if all(count >= n_samples for count in class_counts.values()):
            print(f"All required samples have been saved for {dataset_name}.")
            break

def batch_generate_all_datasets(root_dir: str, output_dir: str, n_samples: int, image_size=(224, 224)):
    """
    Generates N samples per class for all supported datasets.

    Parameters:
        root_dir (str): Directory where datasets will be downloaded.
        output_dir (str): Directory where the images will be saved.
        n_samples (int): Number of samples to save per class.
        image_size (tuple): Size to resize the images to.
    """
    dataset_names = ['CIFAR10', 'CIFAR100', 'FOOD101', 'FLOWERS102']
    for name in dataset_names:
        print(f"\nGenerating samples for dataset: {name}")
        dataset_output_dir = os.path.join(output_dir, name)
        save_n_samples_per_class(
            dataset_name=name,
            root_dir=root_dir,
            output_dir=dataset_output_dir,
            n_samples=n_samples,
            image_size=image_size
        )

if __name__ == "__main__":
    # Configuration
    batch_generate = True
    root = '/media/marronedantas/HD4TB/Projects/datasets'
    output = './output_images'
    samples = 2
    size = (224, 224)

    if batch_generate:
        batch_generate_all_datasets(root, output, samples, size)
    else:
        # Single dataset mode
        save_n_samples_per_class(
            dataset_name='CIFAR10',
            root_dir=root,
            output_dir=output,
            n_samples=samples,
            image_size=size
        )
