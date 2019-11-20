import torch
import torch.nn as nn
import warnings as warn
from .alexnet import *
from .vgg import *
from .resnet import *
from .inception import *
from .mpncovresnet import *
from .densenet import *
from .mpncovvgg import *

def get_basemodel(modeltype, pretrained=False):
    modeltype = globals()[modeltype]
    if pretrained == False:
       warn.warn('You will use model that randomly initialized!')
    return modeltype(pretrained=pretrained)

class Basemodel(nn.Module):
    """Load backbone model and reconstruct it into three part:
       1) feature extractor
       2) global image representaion
       3) classifier
    """
    def __init__(self, modeltype, pretrained=False):
        super(Basemodel, self).__init__()
        basemodel = get_basemodel(modeltype, pretrained)
        self.pretrained = pretrained
        if modeltype.startswith('alexnet'):
            basemodel = self._reconstruct_alexnet(basemodel)
        if modeltype.startswith('vgg'):
            basemodel = self._reconstruct_vgg(basemodel)
        if modeltype.startswith('resnet'):
            basemodel = self._reconstruct_resnet(basemodel)
        if modeltype.startswith('inception'):
            basemodel = self._reconstruct_inception(basemodel)
        if modeltype.startswith('densenet'):
            basemodel = self._reconstruct_densenet(basemodel)
        if modeltype.startswith('mpncovresnet'):
            basemodel = self._reconstruct_mpncovresnet(basemodel) #
        if modeltype.startswith('mpncovvgg'):
            basemodel = self._reconstruct_mpncov_vgg(basemodel)
        self.features = basemodel.features
        self.representation = basemodel.representation
        self.classifier = basemodel.classifier
        self.representation_dim = basemodel.representation_dim
    def _reconstruct_alexnet(self, basemodel):
        model = nn.Module()
        model.features = basemodel.features[:-1]
        model.representation = basemodel.features[-1]
        if self.pretrained:
            model.classifier = basemodel.classifier[-1]
        else:
            model.classifier = basemodel.classifier
        model.representation_dim = 256
        return model
    def _reconstruct_vgg(self, basemodel):
        model = nn.Module()
        model.features = basemodel.features[:-1]
        model.representation = basemodel.features[-1]
        if self.pretrained:
            model.classifier = basemodel.classifier[-1]
        else:
            model.classifier = basemodel.classifier
        model.representation_dim = 512
        return model
    def _reconstruct_resnet(self, basemodel):
        model = nn.Module()
        model.features = nn.Sequential(*list(basemodel.children())[:-2])
        model.representation = basemodel.avgpool
        model.classifier = basemodel.fc
        model.representation_dim=basemodel.fc.weight.size(1)
        return model
    def _reconstruct_inception(self, basemodel):
        model = nn.Module()
        model.features = nn.Sequential(basemodel.Conv2d_1a_3x3,
                                       basemodel.Conv2d_2a_3x3,
                                       basemodel.Conv2d_2b_3x3,
                                       nn.MaxPool2d(kernel_size=3, stride=2),
                                       basemodel.Conv2d_3b_1x1,
                                       basemodel.Conv2d_4a_3x3,
                                       nn.MaxPool2d(kernel_size=3, stride=2),
                                       basemodel.Mixed_5b,
                                       basemodel.Mixed_5c,
                                       basemodel.Mixed_5d,
                                       basemodel.Mixed_6a,
                                       basemodel.Mixed_6b,
                                       basemodel.Mixed_6c,
                                       basemodel.Mixed_6d,
                                       basemodel.Mixed_6e,
                                       basemodel.Mixed_7a,
                                       basemodel.Mixed_7b,
                                       basemodel.Mixed_7c)
        model.representation = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = basemodel.fc
        model.representation_dim=basemodel.fc.weight.size(1)
        return model
    def _reconstruct_densenet(self, basemodel):
        model = nn.Module()
        model.features = basemodel.features
        model.features.add_module('last_relu', nn.ReLU(inplace=True))
        model.representation = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = basemodel.classifier
        model.representation_dim=basemodel.classifier.weight.size(1)
        return model
    def _reconstruct_mpncovresnet(self, basemodel):
        model = nn.Module()
        if self.pretrained:
            model.features = nn.Sequential(*list(basemodel.children())[:-1])
            model.representation_dim=basemodel.layer_reduce.weight.size(0)
        else:
            model.features = nn.Sequential(*list(basemodel.children())[:-4])
            model.representation_dim=basemodel.layer_reduce.weight.size(1)
        model.representation = None
        model.classifier = basemodel.fc
        return model

    def _reconstruct_mpncov_vgg(self, basemodel):
        model = nn.Module()
        model.features = basemodel.features
        model.representation = basemodel.representation
        model.classifier = basemodel.classifier
        model.representation_dim = model.representation.output_dim
        return model
    def forward(self, x):
        x = self.features(x)
        x = self.representation(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out
