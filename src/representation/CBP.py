'''
@file: CBP.py
@author: Chunqiao Xu
@author: Jiangtao Xie
@author: Peihua Li
'''
import torch
import torch.nn as nn
class CBP(nn.Module):
     """Compact Bilinear Pooling
        implementation of Compact Bilinear Pooling (CBP)
        https://arxiv.org/pdf/1511.06062.pdf

     Args:
         thresh: small positive number for computation stability
         projDim: projected dimension
         input_dim: the #channel of input feature
     """
     def __init__(self, thresh=1e-8, projDim=8192, input_dim=512):
         super(CBP, self).__init__()
         self.thresh = thresh
         self.projDim = projDim
         self.input_dim = input_dim
         self.output_dim = projDim
         torch.manual_seed(1)
         self.h_ = [
                 torch.randint(0, self.output_dim, (self.input_dim,),dtype=torch.long),
                 torch.randint(0, self.output_dim, (self.input_dim,),dtype=torch.long)
         ]
         self.weights_ = [
             (2 * torch.randint(0, 2, (self.input_dim,)) - 1).float(),
             (2 * torch.randint(0, 2, (self.input_dim,)) - 1).float()
         ]

         indices1 = torch.cat((torch.arange(input_dim, dtype=torch.long).reshape(1, -1),
                               self.h_[0].reshape(1, -1)), dim=0)
         indices2 = torch.cat((torch.arange(input_dim, dtype=torch.long).reshape(1, -1),
                               self.h_[1].reshape(1, -1)), dim=0)

         self.sparseM = [
             torch.sparse.FloatTensor(indices1, self.weights_[0], torch.Size([self.input_dim, self.output_dim])).to_dense(),
             torch.sparse.FloatTensor(indices2, self.weights_[1], torch.Size([self.input_dim, self.output_dim])).to_dense(),
         ]
     def _signed_sqrt(self, x):
         x = torch.mul(x.sign(), torch.sqrt(x.abs()+self.thresh))
         return x

     def _l2norm(self, x):
         x = nn.functional.normalize(x)
         return x

     def forward(self, x):
         bsn = 1
         batchSize, dim, h, w = x.data.shape
         x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, dim)  # batchsize,h, w, dim,
         y = torch.ones(batchSize, self.output_dim, device=x.device)

         for img in range(batchSize // bsn):
             segLen = bsn * h * w
             upper = batchSize * h * w
             interLarge = torch.arange(img * segLen, min(upper, (img + 1) * segLen), dtype=torch.long)
             interSmall = torch.arange(img * bsn, min(upper, (img + 1) * bsn), dtype=torch.long)
             batch_x = x_flat[interLarge, :]

             sketch1 = batch_x.mm(self.sparseM[0].to(x.device)).unsqueeze(2)
             sketch1 = torch.fft(torch.cat((sketch1, torch.zeros(sketch1.size(), device=x.device)), dim=2), 1)

             sketch2 = batch_x.mm(self.sparseM[1].to(x.device)).unsqueeze(2)
             sketch2 = torch.fft(torch.cat((sketch2, torch.zeros(sketch2.size(), device=x.device)), dim=2), 1)

             Re = sketch1[:, :, 0].mul(sketch2[:, :, 0]) - sketch1[:, :, 1].mul(sketch2[:, :, 1])
             Im = sketch1[:, :, 0].mul(sketch2[:, :, 1]) + sketch1[:, :, 1].mul(sketch2[:, :, 0])

             tmp_y = torch.ifft(torch.cat((Re.unsqueeze(2), Im.unsqueeze(2)), dim=2), 1)[:, :, 0]

             y[interSmall, :] = tmp_y.view(torch.numel(interSmall), h, w, self.output_dim).sum(dim=1).sum(dim=1)

         y = self._signed_sqrt(y)
         y = self._l2norm(y)
         return y
