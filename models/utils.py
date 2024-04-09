import torchvision
import torch.nn as nn
import copy

import torch
import torch.utils.data as data
from torch import Tensor

# get convs from vgg16 with pre-trained weights
def VGG16_initializator():
    layer_names =["conv1_1","conv1_2","conv2_1","conv2_2","conv3_1","conv3_2","conv3_3","conv4_1","conv4_2","conv4_3","conv5_1","conv5_2","conv5_3"] 
    layers = list(torchvision.models.vgg16(pretrained=True).features.children())
    layers = [x for x in layers if isinstance(x, nn.Conv2d)]
    layer_dic = dict(zip(layer_names,layers))
    return layer_dic

# add a CBR-layer from model_dictionary (of vgg16)
def make_crb_layer_from_names(names,model_dic,bn_dim,existing_layer=None):
    layers = []
    if existing_layer is not None:
        layers = [existing_layer,nn.BatchNorm2d(bn_dim,momentum = 0.1),nn.ReLU(inplace=True)]

    for name in names:
        layers += [copy.deepcopy(model_dic[name]), nn.BatchNorm2d(bn_dim,momentum = 0.1), nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)

# add CBR-layers indicating conv filters
def make_crb_layers_from_size(sizes):
    layers = []
    for size in sizes:
        layers += [nn.Conv2d(size[0], size[1], kernel_size=3, padding=1), nn.BatchNorm2d(size[1],momentum = 0.1), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

# count trainable params
def count_parameters(model:nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


