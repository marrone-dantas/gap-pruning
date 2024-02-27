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
    

# Initialize the object detection model
model_path_template = '/media/marrone/M2/Projects/gap_compress/checkpoints/ssd_model_jist_device_device_type_pruned_compression_level.pth'
pruned_level = 1
image_path = '/media/marrone/M2/Projects/gap_compress/download.jpeg'

# With GPU 
detector = ObjectDetectionSSD(model_path_template, pruned_level, True)
detector.detect_objects(image_path)

# Without GPU 
detector = ObjectDetectionSSD(model_path_template, pruned_level, False)
detector.detect_objects(image_path)

