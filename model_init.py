from src.network import *
import torch
import torch.nn as nn
import warnings
__all__ = ['Newmodel', 'get_model']

class Newmodel(Basemodel):
    """replace the image representation method and classifier

       Args:
       modeltype: model archtecture
       representation: image representation method
       num_classes: the number of classes
       freezed_layer: the end of freezed layers in network
       pretrained: whether use pretrained weights or not
    """
    def __init__(self, modeltype, representation, num_classes, freezed_layer, pretrained=False):
        super(Newmodel, self).__init__(modeltype, pretrained)
        if representation is not None:
            representation_method = representation['function']
            representation.pop('function')
            representation_args = representation
            representation_args['input_dim'] = self.representation_dim
            self.representation = representation_method(**representation_args)
            fc_input_dim = self.representation.output_dim
        index_before_freezed_layer = 0
        if freezed_layer:
            for m in self.features.children():
                if index_before_freezed_layer < freezed_layer:
                    m = self._freeze(m)
                index_before_freezed_layer += 1
        if modeltype.startswith('alexnet') or modeltype.startswith('vgg'):
            if not pretrianed:
                self.classifier[0] = nn.Linear(fc_input_dim, 4096)
                self.classifier[-1] = nn.Linear(4096, num_classes)
            else:
                self.classifier = nn.Linear(fc_input_dim, num_classes)
        else:
            self.classifier = nn.Linear(fc_input_dim, num_classes)
    def _freeze(self, modules):
        for param in modules.parameters():
            param.requires_grad = False
        return modules


def get_model(modeltype, representation, num_classes, freezed_layer, pretrained=False):
    _model = Newmodel(modeltype, representation, num_classes, freezed_layer, pretrained=pretrained)
    return _model


