# Fast MPN-COV (i.e., iSQRT-COV)

Created by [Jiangtao Xie](http://jiangtaoxie.github.io) and [Peihua Li](http://www.peihuali.org)
<div>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="http://peihuali.org/pictures/fast_MPN-COV.JPG" width="80%"/>
</div>

## Introduction

This repository contains the source code under **PyTorch** framework and models trained on ImageNet 2012 dataset for the following paper:

         @InProceedings{Li_2018_CVPR,
               author = {Li, Peihua and Xie, Jiangtao and Wang, Qilong and Gao, Zilin},
               title = {Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization},
               booktitle = { IEEE Int. Conf. on Computer Vision and Pattern Recognition (CVPR)},
               month = {June},
               year = {2018}
         }

This paper concerns an iterative matrix square root normalization network (called fast MPN-COV), which is very efficient, fit for large-scale datasets, as opposed to its predecessor (i.e., [MPN-COV](https://github.com/jiangtaoxie/MPN-COV) published in ICCV17) that performs matrix power normalization by Eigen-decompositon. The code on bilinear CNN (B-CNN), compact bilinear pooling and global average pooling etc. is also released for both training from scratch and finetuning. If you use the code, please cite this [fast MPN-COV work](http://peihuali.org/iSQRT-COV/iSQRT-COV_bib.htm)  and its predecessor (i.e., [MPN-COV](http://peihuali.org/MPN-COV/MPN-COV_bib.htm)).

## Classification results

#### Classification results (single crop 224x224, %) on ImageNet 2012 validation set
 <table>
         <tr>
             <th rowspan="2" style="text-align:center;">Network</th>
             <th colspan="2" style="text-align:center;">Top-1 Error</th>
             <th colspan="2" style="text-align:center;">Top-5 Error</th>
             <th colspan="2" style="text-align:center;">Pre-trained models</th>
         </tr>
         <tr>
             <td style="text-align:center;">paper</td>
             <td style="text-align:center;">reproduce</td>
             <td style="text-align:center;">paper</td>
             <td style="text-align:center;">reproduce</td>
             <td style="text-align:center;">GoogleDrive</td>
             <td style="text-align:center;">BaiduCloud</td>
         </tr>
         <tr>
             <td style="text-align:center">fast MPN-COV-VGG-D</td>
             <td style="text-align:center;">26.55</td>
             <td style="text-align:center;"><b>23.98</b></td>
             <td style="text-align:center;">8.94</td>
             <td style="text-align:center;"><b>7.12 </b></td>
             <td style="text-align:center;"><a href="https://drive.google.com/open?id=1oD2QydL8VvK2Zu6Ba5Xe3gFq3to8JsrN">650.4MB</a></td>
             <td style="text-align:center;"><a href="https://pan.baidu.com/s/10DmoGbuHjI_Nsd8bEbefsA">650.4MB</a></td>
         </tr>
         <tr>
             <td style="text-align:center">fast MPN-COV-ResNet50</td>
             <td style="text-align:center;">22.14</td>
             <td style="text-align:center;"><b>21.71</b></td>
             <td style="text-align:center;">6.22</td>
             <td style="text-align:center;"><b>6.13</b></td>
             <td style="text-align:center;"><a href="https://drive.google.com/open?id=19TWen7p9UDyM0Ueu9Gb22NtouR109C6j">217.3MB</a></td>
             <td style="text-align:center;"><a href="https://pan.baidu.com/s/17TxANPJg_j2VyYgXV05OOQ">217.3MB</a></td>
         </tr>
         <tr>
             <td style="text-align:center">fast MPN-COV-ResNet101</td>
             <td style="text-align:center;">21.21</td>
             <td style="text-align:center;"><b>20.99</b></td>
             <td style="text-align:center;">5.68</td>
             <td style="text-align:center;"><b>5.56</b></td>
             <td style="text-align:center;"><a href="https://drive.google.com/open?id=1riur7v3rZ7vnrdj2UZ7EBaTKEGSccYwg">289.9MB</a></td>
             <td style="text-align:center;"><a href="https://pan.baidu.com/s/1_H8MosgzPH0BBmlKw2sr5A">289.9MB</a></td>
         </tr>
</table>

#### Fine-grained classification results (top-1 accuracy rates, %)
<table>
     <tr>
         <th rowspan="2" style="text-align:center;">Backbone model</th>
         <th rowspan="2" style="text-align:center;">Dim.</th>
         <th colspan="2" style="text-align:center;"><a href="http://www.vision.caltech.edu/visipedia/CUB-200-2011.html">Birds</a></th>
         <th colspan="2" style="text-align:center;"><a href="http://ai.stanford.edu/~jkrause/cars/car_dataset.html">Aircrafts</a></th>
         <th colspan="2" style="text-align:center;"><a href="http://www.robots.ox.ac.uk/~vgg/data/oid/">Cars</a></th>
     </tr>
     <tr>
         <td> paper </td>
         <td> reproduce</td>
         <td> paper </td>
         <td> reproduce</td>
         <td> paper </td>
         <td> reproduce</td>
     </tr>
     <tr>
         <td style="text-align:center;">VGG-D</td>
         <td style="text-align:center;">32K</td>
         <td style="text-align:center;">87.2</td>
         <td style="text-align:center;"><b>87.0</b></td>
         <td style="text-align:center;">90.0</td>
         <td style="text-align:center;"><b>91.7</b></td>
         <td style="text-align:center;">92.5</td>
         <td style="text-align:center;"><b>93.2</b></td>
     </tr>
     <tr>
         <td style="text-align:center;">ResNet-50</td>
         <td style="text-align:center;">32K</td>
         <td style="text-align:center;">88.1</td>
         <td style="text-align:center;"><b>88.0</b></td>
         <td style="text-align:center;">90.0</td>
         <td style="text-align:center;"><b>90.3</b></td>
         <td style="text-align:center;">92.8</td>
         <td style="text-align:center;"><b>92.3</b></td>
     </tr>
     <tr>
         <td style="text-align:center;">ResNet-101</td>
         <td style="text-align:center;">32K</td>
         <td style="text-align:center;">88.7</td>
         <td style="text-align:center;">TODO</td>
         <td style="text-align:center;">91.4</td>
         <td style="text-align:center;">TODO</td>
         <td style="text-align:center;">93.3</td>
         <td style="text-align:center;">TODO</td>
     </tr>
</table>

- Our method uses neither bounding boxes nor part annotations
- The reproduced results are obtained by simply finetuning our pre-trained fast MPN-COV-ResNet model with a small learning rate, which do not perform SVM as our paper described.

## Implementation details
We implement our Fast MPN-COV (i.e., iSQRT-COV) [meta-layer](./src/representation/MPNCOV.py) under [PyTorch](https://pytorch.org/) package. Note that though autograd package of PyTorch 0.4.0 or above can compute correctly gradients of our meta-layer, that of PyTorch 0.3.0 fails. As such, we decide to implement the backpropagation of our meta-layer without using autograd package, which works well for both PyTorch release 0.3.0 and 0.4.0.

For making our Fast MPN-COV meta layer can be added in a network conveniently, we reconstruct pytorch official demo [imagenet/](https://github.com/pytorch/examples/tree/master/imagenet) and [models/](https://github.com/pytorch/vision/tree/master/torchvision/models). In which, we divide any network for three parts: 1) features extractor; 2) global image representation; 3) classifier. As such, we can **arbitrarily combine a network with our Fast MPN-COV or some other global image representation methods** (e.g.,Global average pooling, Bilinear pooling, Compact bilinear pooling, etc.) Based on these, we can:

---
> - **Finetune a pre-trained model on any image classification datasets.**
>> AlexNet, VGG, ResNet, Inception, etc.
---
> - **Finetune a pre-trained model with a powerful global image representation method on any image classification datasets.**
>> Fast MPN-COV, Bilinear Pooling (B-CNN), Compact Bilinear Pooling (CBP), etc.
---
> - **Train a model from scratch with a powerful global image representation method on any image classification datasets.**
---
> [Finetune demo](./finetune) and [Train from scratch demo](./trainingFromScratch)



**Welcome to contribution.** In this repository, we will keep updating for containing more networks and global image representation methods.

### Created and Modified

```
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
│   │   ├── CBP.py
│   │   └── Custom.py
│   └── torchviz
│       ├── __init__.py
│       └── dot.py
├── trainingFromScratch
│       └── train.sh
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
3. `pip install -r requirements.txt`
4. prepare the dataset as follows
```
.
├── train
│   ├── class1
│   │   ├── class1_001.jpg
│   │   ├── class1_002.jpg
|   |   └── ...
│   ├── class2
│   ├── class3
│   ├── ...
│   ├── ...
│   └── classN
└── val
    ├── class1
    │   ├── class1_001.jpg
    │   ├── class1_002.jpg
    |   └── ...
    ├── class2
    ├── class3
    ├── ...
    ├── ...
    └── classN
```

#### for training from scracth
1. `cp trainingFromScratch/train.sh ./`
2.  modify the dataset path in `train.sh`
3. `sh train.sh`

#### for finetuning our fast MPN-COV model
1. `cp finetune/finetune.sh ./`
2.  modify the dataset path in `finetune.sh`
3. `sh finetune.sh`

#### for finetuning VGG-model by using BCNN
1. `cp finetune/two_stage_finetune.sh ./`
2.  modify the dataset path in `two_stage_finetune.sh`
3. `sh two_stage_finetune.sh`


## Other Implementations

1. [MatConvNet Implementation](https://github.com/jiangtaoxie/matconvnet.fast-mpn-cov)
2. [TensorFlow Implemention](https://github.com/XuChunqiao/Tensorflow-Fast-MPNCOV)

## Contact

**If you have any questions or suggestions, please contact me**

`jiangtaoxie@mail.dlut.edu.cn`
