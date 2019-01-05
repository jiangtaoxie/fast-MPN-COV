# Fast MPN-COV (i.e., iSQRT-COV)

Created by [Jiangtao Xie](http://jiangtaoxie.github.io) and [Peihua Li](http://www.peihuali.org)
<div>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="http://peihuali.org/pictures/fast_MPN-COV.JPG" width="80%"/>
</div>

## Introduction

This repository contains the source code under PyTorch framework and models trained on ImageNet 2012 dataset for the following paper:

         @InProceedings{Li_2018_CVPR,
               author = {Li, Peihua and Xie, Jiangtao and Wang, Qilong and Gao, Zilin},
               title = {Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization},
               booktitle = { IEEE Int. Conf. on Computer Vision and Pattern Recognition (CVPR)},
               month = {June},
               year = {2018}
         }

In this paper, we propose a fast MPN-COV method for computing matrix square root normalization, which is very efficient, scalable to multiple-GPU configuration, while enjoying matching performance with [MPN-COV](https://github.com/jiangtaoxie/MPN-COV). You can visit our [project page](http://www.peihuali.org/iSQRT-COV) for more details.

## Implementation details
We implement our Fast MPN-COV (i.e., iSQRT-COV) [meta-layer](./src/representation/MPNCOV.py) under [PyTorch](https://pytorch.org/) package. Note that though autograd package of PyTorch 0.4.0 or above can compute correctly gradients of our meta-layer, that of PyTorch 0.3.0 fails. As such, we decide to implement the backpropagation of our meta-layer without using autograd package, which works well for both PyTorch release 0.3.0 and 0.4.0.

For making our Fast MPN-COV meta layer can be added in a network conveniently, we reconstruct pytorch official demo [imagenet/](https://github.com/pytorch/examples/tree/master/imagenet) and [models/](https://github.com/pytorch/vision/torchvision/models). In which, we divide any network for three part: 1) features extractor; 2) global image representation; 3) classifier. As such, we can arbitrarily combine a network with our or some other global image representation methods (e.g.,Global average pooling, Bilinear pooling, Compact bilinear pooling, etc.)

**Call for contributions.** In this repository, we will keep updating for containing more networks and global image representation methods.

- [x] Matrix power normalized cov pooling (MPNCOV)
- [x] Bilinear CNN (BCNN)
- [ ] Compact Bilinear pooling (CBP)
- [ ] etc.

### Created and Modified

```
.
├── main.py
├── imagepreprocess.py
├── functions.py
├── model_init.py
├── src
│   ├── network
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── inception.py
│   │   ├── alexnet.py
│   │   ├── mpncovresnet.py
│   │   ├── resnet.py
│   │   └── vgg.py
│   ├── representation
│   │   ├── __init__.py
│   │   ├── MPNCOV.py
│   │   ├── GAvP.py
│   │   ├── BCNN.py
│   │   └── Custom.py
│   └── torchviz
│       ├── __init__.py
│       └── dot.py
├── trainingFromScratch
│    └── train.sh
└── finetune
        ├── finetune.sh
        └── two_stage_finetune.sh
```
##### For more convenient training and finetuning, we

-  implement some functions for plotting convergence curve.  
-  adopt network visualization tool [pytorchviz](https://github.com/szagoruyko/pytorchviz) for plotting network structure.
- use shell file to manage the process.

## Installation and Usage

1. Install [PyTorch](https://github.com/pytorch/pytorch) (0.4.0 or above)
2. type `git clone https://github.com/jiangtaoxie/fast-MPN-COV`
3. `pip install -r requirement.txt`

#### for training from scracth
1. `cp trainingFromScratch/train.sh ./train.sh`
2.  modify the dataset path in `train.sh`
3. `sh train.sh`

#### for finetuning our fast MPN-COV model
1. `cp finetune/finetune.sh ./finetune.sh`
2.  modify the dataset path in `finetune.sh`
3. `sh finetune.sh`

#### for finetuning VGG-model by using BCNN
1. `cp finetune/two_stage_finetune.sh ./two_stage_finetune.sh`
2.  modify the dataset path in `two_stage_finetune.sh`
3. `sh two_stage_finetune.sh`

## Classification results

### Classification results (single crop 224x224, %) on ImageNet 2012 validation set
 <table>
         <tr>
             <th rowspan="2" style="text-align:center;">Package</th>
             <th rowspan="2" style="text-align:center;">Network</th>
             <th rowspan="2" style="text-align:center;">Top-1 Error</th>
             <th rowspan="2" style="text-align:center;">Top-5 Error</th>
             <th colspan="2" style="text-align:center;">Pre-trained models</th>
         </tr>
         <tr>
             <td style="text-align:center;">GoogleDrive</td>
             <td style="text-align:center;">BaiduCloud</td>
         </tr>
         <tr>
             <td rowspan="4" style="text-align:center">PyTorch</td>
             <td>fast MPN-COV-ResNet50</td>
             <td style="text-align:center;">21.71</td>
             <td style="text-align:center;">6.13</td>
             <td style="text-align:center;"><a href="https://drive.google.com/file/d/132PzY3eVDuGg8ROz5wON5FTC2E2o12Ck/view?usp=sharing">217.3MB</a></td>
             <td style="text-align:center;"><a href="https://pan.baidu.com/s/16TsQWwY7iWrvwTE43dwgHQ">217.3MB</a></td>
         </tr>
         <tr>
             <td style="text-align:center">fast MPN-COV-ResNet101</td>
             <td colspan="4" style="text-align:center;">coming soon</td>
         </tr>
         <tr>
             <td style="text-align:center">ResNet50</td>
             <td style="text-align:center;">23.85</td>
             <td style="text-align:center;">7.13</td>
             <td colspan="2" rowspan="2" style="text-align:center;">results from
             <a href="https://pytorch.org/docs/stable/torchvision/models.html">PyTorch</a></td>
         </tr>
         <tr>
             <td style="text-align:center">ResNet101</td>
             <td style="text-align:center;">22.63</td>
             <td style="text-align:center;">6.44</td>
         </tr>
         <tr>
             <td rowspan="4" style="text-align:center">MatConvNet</td>
             <td style="text-align:center">fast MPN-COV-ResNet50</td>
             <td style="text-align:center;">22.14</td>
             <td style="text-align:center;">6.22</td>
             <td style="text-align:center;"><a href="https://drive.google.com/open?id=1fG5Mz6GzlMt7TeWq_HAr7NVqetVpgrRS">202.7MB</a></td>
             <td style="text-align:center;"><a href="https://pan.baidu.com/s/1I1XvWfx8JGB02OUHCxXpEg">202.7MB</a></td>
         </tr>
         <tr>
             <td style="text-align:center">fast MPN-COV-ResNet101</td>
             <td style="text-align:center;">21.21</td>
             <td style="text-align:center;">5.68</td>
             <td style="text-align:center;"><a href="https://drive.google.com/open?id=1ezNfxAcZNuWChIkjjC1eabVdNuVwObbS">270.5MB</a></td>
             <td style="text-align:center;"><a href="https://pan.baidu.com/s/1YuETiWAfw-RGN0sVxDlU8g">270.5MB</a></td>
         </tr>
         <tr>
             <td style="text-align:center">ResNet50</td>
             <td style="text-align:center;">24.6</td>
             <td style="text-align:center;">7.7</td>
             <td colspan="2" rowspan="2" style="text-align:center;">results from
             <a href="http://www.vlfeat.org/matconvnet/pretrained/">MatConvNet</a></td>
         </tr>
         <tr>
             <td style="text-align:center">ResNet101</td>
             <td style="text-align:center;">23.4</td>
             <td style="text-align:center;">7.0</td>
         </tr>
</table>


### Fine-grained classification results (top-1 accuracy rates, %)
<table>
     <tr>
         <th style="text-align:center;"></th>
         <th style="text-align:center;">Method</th>
         <th style="text-align:center;">Dimension</th>
         <th style="text-align:center;"><a href="http://www.vision.caltech.edu/visipedia/CUB-200-2011.html">Birds</a><br/><img src="http://jtxie.com/images/bird.jpg" width="50px"></th>
         <th style="text-align:center;"><a href="http://ai.stanford.edu/~jkrause/cars/car_dataset.html">Aircrafts</a><br/><img src="http://jtxie.com/images/aircraft.jpeg" width="50px"></th>
         <th style="text-align:center;"><a href="http://www.robots.ox.ac.uk/~vgg/data/oid/">Cars</a><br/><img src="http://jtxie.com/images/cars.jpg" width="50px"></th>
     </tr>
     <tr>
         <td rowspan="4" style="text-align:center;">ResNet-50</td>
         <td rowspan="2" style="text-align:center;">fast MPN-COV</td>
         <td style="text-align:center;">32K</td>
         <td style="text-align:center;"><b>88.1</b></td>
         <td style="text-align:center;"><b>90.0</b></td>
         <td style="text-align:center;"><b>92.8</b></td>
     </tr>
     <tr>
         <td style="text-align:center;">8K</td>
         <td style="text-align:center;">87.3</td>
         <td style="text-align:center;">89.5</td>
         <td style="text-align:center;">91.7</td>
     </tr>
     <tr>
         <td style="text-align:center;">CBP</td>
         <td style="text-align:center;">14K</td>
         <td style="text-align:center;">81.6</td>
         <td style="text-align:center;">81.6</td>
         <td style="text-align:center;">88.6</td>
     </tr>
     <tr>
         <td style="text-align:center;">KP</td>
         <td style="text-align:center;">14K</td>
         <td style="text-align:center;">84.7</td>
         <td style="text-align:center;">85.7</td>
         <td style="text-align:center;">91.1</td>
     </tr>
     <tr>
         <td rowspan="8" style="text-align:center;">VGG-D</td>
         <td style="text-align:center;">fast MPN-COV</td>
         <td style="text-align:center;">32K</td>
         <td style="text-align:center;"><b>87.2</b></td>
         <td style="text-align:center;"><b>90.0</b></td>
         <td style="text-align:center;"><b>92.5</b></td>
     </tr>
     <tr>
         <td style="text-align:center;">NetVLAD</td>
         <td style="text-align:center;">32K</td>
         <td style="text-align:center;">81.9</td>
         <td style="text-align:center;">81.8</td>
         <td style="text-align:center;">88.6</td>
     </tr>
     <tr>
         <td style="text-align:center;">CBP</td>
         <td style="text-align:center;">8K</td>
         <td style="text-align:center;">84.3</td>
         <td style="text-align:center;">84.1</td>
         <td style="text-align:center;">91.2</td>
     </tr>
     <tr>
         <td style="text-align:center;">KP</td>
         <td style="text-align:center;">13K</td>
         <td style="text-align:center;">86.2</td>
         <td style="text-align:center;">86.9</td>
         <td style="text-align:center;">92.4</td>
     </tr>
     <tr>
         <td style="text-align:center;">LRBP</td>
         <td style="text-align:center;">10K</td>
         <td style="text-align:center;">84.2</td>
         <td style="text-align:center;">87.3</td>
         <td style="text-align:center;">90.9</td>
     </tr>
     <tr>
         <td style="text-align:center;">Impro. B-CNN</td>
         <td style="text-align:center;">262K</td>
         <td style="text-align:center;">85.8</td>
         <td style="text-align:center;">88.5</td>
         <td style="text-align:center;">92.0</td>
     </tr>
     <tr>
         <td style="text-align:center;">G2DeNet</td>
         <td style="text-align:center;">263K</td>
         <td style="text-align:center;">87.1</td>
         <td style="text-align:center;">89.0</td>
         <td style="text-align:center;">92.5</td>
     </tr>
     <tr>
         <td style="text-align:center;">HIHCA</td>
         <td style="text-align:center;">9KK</td>
         <td style="text-align:center;">85.3</td>
         <td style="text-align:center;">88.3</td>
         <td style="text-align:center;">91.7</td>
     </tr>
     <tr>
         <td colspan="2" style="text-align:center;"> fast MPN-COV with ResNet-101</td>
         <td style="text-align:center;">32K</td>
         <td style="text-align:center;"><b>88.7</b></td>
         <td style="text-align:center;"><b>91.4</b></td>
         <td style="text-align:center;"><b>93.3</b></td>
     </tr>
</table>

- Our method uses neither bounding boxes nor part annotations

## Other Implementations

1. [MatConvNet Implementation](https://github.com/jiangtaoxie/matconvnet.fast-mpn-cov)
2. [TensorFlow Implemention](./TensorFlow)(coming soon)

## Contact

**If you have any questions or suggestions, please contact me**

`jiangtaoxie@mail.dlut.edu.cn`
