import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_full_model(path):
    model = torch.load(path, map_location=torch.device('cpu'), weights_only=False)    
    return model

def calculate_global_pruning(original_model_path, pruned_model_path):
    original_model = load_full_model(original_model_path)
    pruned_model = load_full_model(pruned_model_path)

    original_params = count_parameters(original_model)
    pruned_params = count_parameters(pruned_model)

    pruning_ratio = (1 - pruned_params / original_params) * 100
    compression_factor = original_params / pruned_params if pruned_params > 0 else float('inf')

    print(f"\n🔍 Comparação entre modelos:")
    print(f"Original model parameters : {original_params:,}")
    print(f"Pruned model parameters   : {pruned_params:,}")
    print(f"Global pruning rate       : {pruning_ratio:.2f}%")
    print(f"Compression factor        : {compression_factor:.2f}x smaller")

    return pruning_ratio, compression_factor

if __name__ == "__main__":
    # Define os caminhos aqui
    base_path = '/media/marronedantas/HD4TB/Projects/gap-pruning/checkpoints/'

    original_model_path = base_path + 'acc_full_model_vgg_cifar10.pt'
    pruned_model_path   = base_path + 'acc_full_model_pruned_90_vgg_cifar10.pt'

    calculate_global_pruning(original_model_path, pruned_model_path)
