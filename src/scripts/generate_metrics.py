from torch import jit
import torchvision.transforms as T
from PIL import Image, ImageDraw
import torch
import time
import cv2
import numpy as np
from torchinfo import summary
import os
import pandas as pd
import torch

class ObjectDetectionSSD:
    def __init__(self, model_path_template, compression_level, use_gpu=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.model_path_template = model_path_template
        self.model_path = None
        self.compression_level = compression_level
        self.model = self.load_model()
        self.transform = T.Compose([
            T.Resize(300),  # Assuming SSD300
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        # Corrected string formatting to use named placeholders
        device_str = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.model_path = self.model_path_template.replace("device_type", device_str).replace("compression_level", str(self.compression_level))
        print(f"Loading model from {self.model_path}")
        model = jit.load(self.model_path).to(self.device)
        model.eval()
        return model

    def preprocess_image(self, image_input):
        if isinstance(image_input, np.ndarray):
            # Convert numpy array (cv2 image) to PIL Image
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        elif isinstance(image_input, str):
            # Load image from file path
            image = Image.open(image_input).convert("RGB")
        else:
            raise ValueError("Unsupported image input type")
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image, input_tensor

    def detect_objects(self, image_input, total_predictions=1):
        image, input_tensor = self.preprocess_image(image_input)
        predictions, infer_times = self.perform_inference(input_tensor, total_predictions)
        detections = self.process_predictions(predictions, image)
        return image, detections, infer_times

    def perform_inference(self, input_tensor, total_predictions):
        infer_times = []
        with torch.no_grad():
            
            for _ in range(total_predictions):
                start_time = time.time()
                predictions = self.model(input_tensor)
                end_time = time.time()
                inference_time = (end_time - start_time) 
                infer_times.append(inference_time)
                
        return [k.to('cpu') for k in predictions], infer_times

    def process_predictions(self, predictions, image):
        draw = ImageDraw.Draw(image)
        detected_objects = []
        classes = ['background', 'class1', 'class2']
        for i, (box, label, score) in enumerate(zip(predictions[0], predictions[1], predictions[2])):
            if score >= 0.5:
                box_coordinates = box.tolist()
                label_text = classes[label]
                draw.rectangle(box_coordinates, outline="red", width=3)
                draw.text((box_coordinates[0], box_coordinates[1]), f"{label_text}: {score:.2f}", fill="red")
                detected_objects.append({
                    'coordinates': box_coordinates,
                    'label': label_text,
                    'score': score.item()
                })
        return detected_objects
    
    def get_power_consumption_estimate(self, average_time_hours, full_load_power_watts):

        # Calculate estimated power consumption
        estimated_consumption_watts = average_time_hours * full_load_power_watts
        return estimated_consumption_watts
    
    def get_metrics(self, image_input, total_predictions=1, full_load_power_watts=100.0, aditional_parr=None):
        
        #Get the image, the detections and the inference time
        _, _, infer_times = self.detect_objects(image_input, total_predictions)

        # Calculate the average inference time
        # Define quantiles for outlier removal
        lower_quantile = 0.10  # Remove lower 10%
        upper_quantile = 0.90  # Remove upper 10%

        # Calculate quantile values again with numpy imported
        lower_bound = np.quantile(infer_times, lower_quantile)
        upper_bound = np.quantile(infer_times, upper_quantile)

        # Filter the list to remove values outside the quantiles
        filtered_times = [time for time in infer_times if lower_bound <= time <= upper_bound]

        # Calculate the average of the filtered list
        inference_time = sum(filtered_times) / len(filtered_times)
        
        # Memory usage (RAM) and model summary
        model_info = summary(self.model, input_data=self.transform(Image.new('RGB', (300, 300))).unsqueeze(0).to(self.device), verbose=0)

        #Generating the metrics for memory usage, disk usage, cpu occupancy and power consumption
        total_params = model_info.total_params
        total_output = model_info.total_input
        memory_usage = (total_params + total_output) * 4 / (1024 ** 2)  # Assuming 4 bytes per float, convert to MB
        
        # Disk usage (size of model file)
        disk_usage = os.path.getsize(self.model_path) / (1024 ** 2)  # Convert bytes to MB

        # Power consumption approximation
        power_consumption = self.get_power_consumption_estimate(inference_time/3600.0, full_load_power_watts)

        metrics = {
            'memory_usage_MB': memory_usage,
            'disk_usage_MB': disk_usage,
            'average_inference_time_seconds': inference_time,
            'estimated_power_consumption': power_consumption,
            'fps': 1/inference_time
        }
        
        if aditional_parr is not None:
            metrics[aditional_parr[0]] = aditional_parr[1]
        
        return metrics




# Initialize the object detection model
model_path_template = '/media/marrone/M2/Projects/gap_compress/checkpoints/ssd_model_jist_device_device_type_pruned_compression_level.pth'
pruned_level = [1,10,20,80,90,99]
accuracy_list = [0.97, 0.95, 0.91, 0.84, 0.82, 0.80]
image_path = '/media/marrone/M2/Projects/gap_compress/download.jpeg'
metrics_dict = {}
total_predictions = 200

# Save default PyTorch threads settings
default_num_threads = torch.get_num_threads()
default_num_interop_threads = torch.get_num_interop_threads()

for k, level in enumerate(pruned_level):
    # With GPU (RTX 3060 = 170W)
    detector = ObjectDetectionSSD(model_path_template, level, True)
    partial_metrics = detector.get_metrics(image_path, total_predictions=total_predictions, full_load_power_watts=170.0, aditional_parr=['acc', accuracy_list[k]])
    metrics_dict['GPU_'+str(level)] = partial_metrics

    # Without GPU (CPU = 65W)
    detector = ObjectDetectionSSD(model_path_template, level, False)
    partial_metrics = detector.get_metrics(image_path, total_predictions=total_predictions, full_load_power_watts=65.0, aditional_parr=['acc', accuracy_list[k]])
    metrics_dict['CPU_'+str(level)] = partial_metrics
    
    # Convert the nested dictionary into a DataFrame
    df_metrics = pd.DataFrame.from_dict(metrics_dict, orient='index')

    # Optionally, reset the index to make the configuration a column instead of the index
    df_metrics.reset_index(inplace=True)
    df_metrics.rename(columns={'index': 'configuration'}, inplace=True)

    # Display the DataFrame to verify its structure
    df_metrics.to_csv('metrics.csv', index=False)
    
#Runing the model for the contrained scenarios

# Simulated Raspberry Pi scenario (Cortex-A72 quad-core 1,5GHz)
# Assuming Raspberry Pi's power consumption at full load is 15W
# Limit PyTorch to use fewer threads to simulate a less powerful CPU
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
metrics_dict = {}

for k, level in enumerate(pruned_level):
        
    detector = ObjectDetectionSSD(model_path_template, level, False)
    partial_metrics = detector.get_metrics(image_path, total_predictions=total_predictions, full_load_power_watts=15.0, aditional_parr=['acc', accuracy_list[k]])
    metrics_dict['RaspberryPi_'+str(level)] = partial_metrics
    
    # Convert the nested dictionary into a DataFrame
    df_metrics = pd.DataFrame.from_dict(metrics_dict, orient='index')

    # Optionally, reset the index to make the configuration a column instead of the index
    df_metrics.reset_index(inplace=True)
    df_metrics.rename(columns={'index': 'configuration'}, inplace=True)

    # Display the DataFrame to verify its structure
    df_metrics.to_csv('metrics_rasp.csv', index=False)


