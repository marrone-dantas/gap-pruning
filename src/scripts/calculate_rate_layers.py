import torch

def load_full_model(path):
    return torch.load(path, map_location=torch.device('cpu'), weights_only=False)

def count_parameters_recursive(model, prefix=''):
    layer_params = {}
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # Se o módulo não tiver filhos, ele é uma "folha"
        if len(list(module.children())) == 0:
            total = sum(p.numel() for p in module.parameters() if p.requires_grad)
            layer_params[full_name] = total
        else:
            # Chamada recursiva
            sub_params = count_parameters_recursive(module, full_name)
            layer_params.update(sub_params)

    return layer_params

def calculate_recursive_pruning(original_path, pruned_path):
    original_model = load_full_model(original_path)
    pruned_model = load_full_model(pruned_path)

    original_layers = count_parameters_recursive(original_model)
    pruned_layers = count_parameters_recursive(pruned_model)

    total_layers = len(original_layers)
    pruned_layer_count = 0

    print(f"{'Layer':60s} | {'Original':>10} | {'Pruned':>10} | {'Delta':>10}")
    print('-' * 95)

    for name in original_layers:
        orig = original_layers.get(name, 0)
        prun = pruned_layers.get(name, 0)
        delta = orig - prun
        if delta > 0:
            pruned_layer_count += 1
        print(f"{name:60s} | {orig:10d} | {prun:10d} | {delta:10d}")

    pruning_layer_ratio = (pruned_layer_count / total_layers) * 100
    print(f"\nTotal layers: {total_layers}")
    print(f"Pruned layers: {pruned_layer_count}")
    print(f"Pruning per layer: {pruning_layer_ratio:.2f}%")

    return pruning_layer_ratio

if __name__ == "__main__":
    base_path = '/media/marronedantas/HD4TB/Projects/gap-pruning/checkpoints/'
    
    original_model_path = base_path + 'acc_full_model_vgg_cifar10.pt'
    pruned_model_path   = base_path + 'acc_full_model_pruned_30_vgg_cifar10.pt'

    calculate_recursive_pruning(original_model_path, pruned_model_path)
