'''
@file: BCNN.py
@author: Jiangtao Xie
@author: Peihua Li
'''
import torch
import torch.nn as nn

class BCNN(nn.Module):
     """Bilinear Pool
        implementation of Bilinear CNN (BCNN)
        https://arxiv.org/abs/1504.07889v5

     Args:
         thresh: small positive number for computation stability
         is_vec: whether the output is a vector or not
         input_dim: the #channel of input feature
     """
     def __init__(self, thresh=1e-8, is_vec = True, input_dim=2048):
         super(BCNN, self).__init__()
         self.thresh = thresh
         self.is_vec = is_vec
         self.output_dim = input_dim * input_dim
     def _bilinearpool(self, x):
         batchSize, dim, h, w = x.data.shape
         x = x.reshape(batchSize, dim, h * w)
         x = 1. / (h * w) * x.bmm(x.transpose(1, 2))
         return x

     def _signed_sqrt(self, x):
         x = torch.mul(x.sign(), torch.sqrt(x.abs()+self.thresh))
         return x

     def _l2norm(self, x):
         x = nn.functional.normalize(x)
         return x

     def forward(self, x):
         x = self._bilinearpool(x)
         x = self._signed_sqrt(x)
         if self.is_vec:
             x = x.view(x.size(0),-1)
         x = self._l2norm(x)
         return x
