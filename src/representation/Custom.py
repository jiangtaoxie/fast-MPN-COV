import torch
import torch.nn as nn

class Custom(nn.Module):
     def __init__(self, input_dim=2048):
         super(Custom, self).__init__()
         #self.function = 
         self.output_dim = input_dim

     def forward(self, x):
         # x = self.function(x)
         return x
