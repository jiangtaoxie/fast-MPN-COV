## Implementation details
We implement our Fast MPN-COV (i.e., iSQRT-COV) meta-layer under [PyTorch](https://pytorch.org/) package. We design three individual layers: 1) Covpool Layer, estimating global covariance matrix given convolutional features; 2) Sqrtm Layer, which computes matrix square root of any SPD matrix by pre-normalization, Newton-Schulz iteration and post-compensation; 3) Triuvec Layer, concatenating the upper triangular part of a full SPD matrix. Note that though autograd package of PyTorch 0.4.0 can compute correctly gradients of our meta-layer, that of PyTorch 0.3.0 fails. As such, we decide to implement the backpropagation of our meta-layer without using autograd package, which works well for both PyTorch release 0.3.0 and 0.4.0.

### Created and Modified

1. Files we created to implement fast MPN-COV meta-layer
```
├── MPNCOV
    ├── __init__.py
    └── python
        ├── __init__.py
        └── MPNCOV.py
```
 - In this project, we only implement pre-norm. and post-com. by trace normalization.
2. Files we modified to support Fast MPN-COV meta-layer
```
├── main.py
└── vision
    └── torchvision
        └── models
            └── resnet.py
```
 - We duplicated `vision/` and `main.py` from [pytorch/vision/](https://github.com/pytorch/vision) and [pytorch/examples/imagenet/](https://github.com/pytorch/examples/imagenet), respectively. And added our Fast MPN-COV meta-layer in `resnet.py`

## Installation

1. Install [PyTorch](https://github.com/pytorch/pytorch) (0.4.0 or above)
2. Download this repository.
3. `cd vision/` and `sudo python setup.py install`
3. Follow the [introduction](https://github.com/pytorch/examples/imagenet/README.md) of Pytorch to train a MPN-COV-ResNet model.

## Usage
### Insert fast MPN-COV layer into your network
1. Import our Fast MPN-COV package

```python
from MPNCOV.python import MPNCOV
```  
2. Add layers to your network

```python
x = MPNCOV.CovpoolLayer(x) # Global Covariance Pooling layer
x = MPNCOV.SqrtmLayer(x, 5) # Matrix sqaure root layer (including pre-norm., Newton-Schulz iter. and post-com.)
                            # 5 iterations
y = MPNCOV.TriuvecLayer(x) # Layer for concatenation of upper triangular entries
```
