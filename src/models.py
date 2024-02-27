import torchvision.models as models
import torch.nn as nn
from torchsummary import summary

from custom_models.TinyVGG import TinyVGG

def get_model(name_model,num_classes):

    model = None

    if (name_model=='resnet'):
        
        model = models.resnet18(pretrained=True)

        model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    if (name_model=='tinyvgg'):

        model = TinyVGG(num_classes=num_classes)

    if (name_model=='alexnet'):

        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(4096,num_classes)

    if (name_model=='vgg'):
        
        model =  models.vgg16(pretrained=True)   
        model.classifier[6] = nn.Linear(4096,num_classes)
        
    return model
