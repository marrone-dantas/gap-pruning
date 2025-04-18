import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as transF
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch_pruning as tp
import torchvision.models as models
import os

class GapPruning:
    def __init__(self, model, dataset, batch_size=1, device='cuda'):
        self.model = model.to(device)
        self.pruned_model = None
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.global_gap_outputs = {}
        self.feature_maps = {} 
        self.std_devs = {}
        self.pruning_proposal = {}
        self.current_labels = None
        self.hook_handles = []
        
    class HookFunction:
        def __init__(self, parent, layer_name):
            self.parent = parent
            self.layer_name = layer_name

        def __call__(self, module, input, output):
            gap_output = self.parent.global_avg_pool(output)
            self.parent.global_gap_outputs[self.layer_name].append(gap_output.cpu().detach())


    def register_hook_for_conv_layers(self):
        self.hook_handles = []  # Add this line to store hook handles
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.global_gap_outputs[name] = []
                hook_fn = self.HookFunction(self, name)
                handle = module.register_forward_hook(hook_fn)  # Save the handle
                self.hook_handles.append(handle)

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []  

    
    def hook_fn(self, layer_name):
        def hook(module, input, output):
            gap_output = self.global_avg_pool(output)
            self.global_gap_outputs[layer_name].append(gap_output.cpu().detach())
        return hook

    def global_avg_pool(self, tensor):
        return F.adaptive_avg_pool2d(tensor, (1, 1)).view(tensor.size(0), -1)

    def process_dataset(self, samples_size=None, to_cpu=False):
        
        if isinstance(self.dataset, DataLoader):
            data_loader = self.dataset
        else:
            data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        
        self.register_hook_for_conv_layers()
        count_samples = 0
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc="Processing Dataset", total=len(data_loader)):
                if (to_cpu):
                    inputs = inputs.to('cpu')
                else:
                    inputs = inputs.to(self.device)
                self.current_labels = labels.cpu().numpy()  # Store current batch labels
                self.model(inputs)
                
                count_samples += 1
                
                if samples_size is not None and count_samples >= samples_size:
                    break

    def compute_std_devs(self, file_path='std_devs.pth', flg_load=False):
        warnings = []  # List to accumulate warnings

        if os.path.isfile(file_path) and flg_load:
            self.std_devs = torch.load(file_path)
        else:
            for layer_name, outputs in tqdm(self.global_gap_outputs.items(), desc="Computing Std Devs"):
                std_devs_per_output = []

                for output in outputs:
                    std_dev_per_tensor = torch.std(output, dim=0)
                    std_devs_per_output.append(std_dev_per_tensor)

                if std_devs_per_output:
                    self.std_devs[layer_name] = torch.mean(torch.stack(std_devs_per_output), dim=0)
                else:
                    warnings.append(f"Warning: No outputs captured for layer {layer_name}, skipping.")

            torch.save(self.std_devs, file_path)

        # Display warnings at the end of the process, if any
        if warnings:
            print("Some warnings were generated during std_dev computation: No outputs captured for layers:")


    def generate_pruning_proposal(self, pruning_percentage):
        for layer_name, std_dev in self.std_devs.items():
            total_filters = len(std_dev)  # Total number of filters in the layer
            # Calculate the number of filters to prune based on the percentage
            num_filters_to_prune = int(total_filters * pruning_percentage / 100)

            # Ensure at least one filter remains
            num_filters_to_prune = min(num_filters_to_prune, total_filters - 1)

            # Get indices of filters sorted by their standard deviation (lowest first)
            sorted_indices = torch.argsort(std_dev)

            # Select the top-ranked filters for pruning, ensuring at least one filter remains
            filters_to_prune = sorted_indices[:num_filters_to_prune].tolist()

            # Save the pruning proposal for this layer
            self.pruning_proposal[layer_name] = filters_to_prune


    def prune(self):
        DG = tp.DependencyGraph().build_dependency(self.model, example_inputs=torch.randn(1, 3, 224, 224))
        for layer_name, filters_to_prune in self.pruning_proposal.items():
            layer = self.get_layer_by_name(layer_name)
            if layer:
                pruning_plan = DG.get_pruning_plan(layer, tp.prune_conv, idxs=filters_to_prune)
                pruning_plan.exec()

    def get_layer_by_name(self, layer_name):
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        return None
    
    def quantize_model(self):
        # Ensure the model is fully transferred to the CPU.
        self.model.to('cpu')

        # Apply quantization configuration
        quantization_config = torch.quantization.get_default_qconfig('fbgemm')
        self.model.qconfig = quantization_config

        # Prepare the model for quantization
        torch.quantization.prepare(self.model, inplace=True)

        # Modify the `process_dataset` method or ensure it processes data on CPU
        # It's crucial that during calibration, the model and data are on the same device (CPU)
        self.process_dataset(samples_size=10, to_cpu=True)  # Adjust samples_size as needed

        # Convert the prepared model to a quantized version
        torch.quantization.convert(self.model, inplace=True)

        return self.model





