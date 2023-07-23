import torch

from diffusion import DiffusionModel
from unet import UnetModel

import torch.onnx

input_tensor = torch.randn(2, 3, 64, 64)
unet = UnetModel(3, 3)
torch.onnx.export(unet, (input_tensor, torch.randn(2)), "unet.onnx")
