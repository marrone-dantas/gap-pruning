import models as models
import torch
import dataset as dataset

from ai_edge_torch.quantize import pt2e_quantizer
from ai_edge_torch.quantize import quant_config
from torch.ao.quantization import quantize_pt2e

import cv2
import numpy

import ai_edge_torch


model = torch.load('/media/marronedantas/HD4TB/Projects/gap-pruning/checkpoints/acc_full_model_pruned_85_alexnet_cifar10.pt', weights_only=False)

model.to('cpu')
model.eval()

sample_inputs = (torch.randn(1, 3, 224, 224),)
torch_output = model(*sample_inputs)


quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
    pt2e_quantizer.get_symmetric_quantization_config()
)

model = torch._export.capture_pre_autograd_graph(model, sample_inputs)
model = quantize_pt2e.prepare_pt2e(model, quantizer)


img = cv2.imread("/media/marronedantas/HD4TB/Projects/gap-pruning/sample/images.jpeg")
img = cv2.resize(img, (224, 224))
img = numpy.expand_dims(img, 0)
img = numpy.transpose(img, (0, 3, 1, 2))
img = img / 255.0
img = img.astype(numpy.float32)

model(torch.from_numpy(img)) 

model = quantize_pt2e.convert_pt2e(model, fold_quantize=False)

with_quantizer = ai_edge_torch.convert(
    model,
    sample_inputs,
    quant_config=quant_config.QuantConfig(pt2e_quantizer=quantizer),
)
with_quantizer.export("pruned_model_int8.tflite")
