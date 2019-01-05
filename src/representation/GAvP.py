import torch
import torch.nn as nn

class GAvP(nn.Module):
     """Global Average pooling
        Widely used in ResNet, Inception, DenseNet, etc.
     """
     def __init__(self, input_dim=2048):
         super(GAvP, self).__init__()
         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
         self.output_dim = input_dim

     def forward(self, x):
         x = self.avgpool(x)
         return x
