## Demo code on training fast MPN-COV ConvNets from scratch on ImageNet

This demo contains the code which implements training our fast MPN-COV ConvNets from scratch on ImageNet 2012.

### Usage

#### Tutorial

1. Download the ImageNet 2012 [dataset](http://image-net.org/download.php).

2. Put the dataset you downloaded into `matconvnet_root_dir/data/`.

3. Modify the `opts.modelType` in `fast_MPN_COV_main.m` to your needs.

4. run `fast_MPN_COV_main.m`.

#### Function descriptions

1. `fast_MPN_COV_main.m`: The main function.

2. `MPN_COV_init_resnet.m`: Initialize, for training from scratch, the proposed fast MPN-COV ConvNets under ResNet architecture using DagNN, and set the hyper-parameters involved in training.

3. `MPN_COV_init_simplenn.m`: Initialize, for training from scratch, the proposed fast MPN-COV ConvNets under the architectures of AlexNet, VGG-M and VGG-VD using SimpleNN, and set the hyper-parameters involved in training.
