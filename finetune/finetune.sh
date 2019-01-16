set -e
:<<!
*****************Instruction*****************
Here you can easily creat a model by selecting
an arbitray backbone model and global method.
You can fine-tune it on your own datasets by
using a pre-trained model.
Modify the following settings as you wish !
*********************************************
!

#***************Backbone model****************
#Our code provides some mainstream architectures:
#alexnet
#vgg family:vgg11, vgg11_bn, vgg13, vgg13_bn,
#           vgg16, vgg16_bn, vgg19_bn, vgg19
#resnet family: resnet18, resnet34, resnet50,
#               resnet101, resnet152
#mpncovresnet: mpncovresnet50, mpncovresnet101
#inceptionv3
#You can also add your own network in src/network
arch=mpncovresnet50
#*********************************************

#***************global method****************
#Our code provides some global method at the end
#of network:
#GAvP (global average pooling),
#MPNCOV (matrix power normalized cov pooling),
#BCNN (bilinear pooling)
#CBP (compact bilinear pooling)
#...
#You can also add your own method in src/representation
image_representation=MPNCOV
# short description of method
description=reproduce
#*********************************************

#*******************Dataset*******************
#Choose the dataset folder
benchmark=Dataset Name
datadir=/path/to/the/data
dataset=$datadir/$benchmark
num_classes=#classes
#*********************************************

#****************Hyper-parameters*************

# Freeze the layers before a certain layer.
freeze_layer=0
# Batch size
batchsize=10
# The number of total epochs for training
epoch=100
# The inital learning rate
# decreased by step method
lr=1.2e-3
lr_method=step
lr_params=100
# log method
# description: lr = logspace(params1, params2, #epoch)

#lr_method=log
#lr_params=-1.1\ -5.0
weight_decay=1e-3
classifier_factor=5
#*********************************************
echo "Start finetuning!"
modeldir=Results/Finetune-$benchmark-$arch-$image_representation-$description-lr$lr-bs$batchsize
if [ ! -d  "Results" ]; then

mkdir Results

fi
if [ ! -e $modeldir/*.pth.tar ]; then

if [ ! -d  "$modeldir" ]; then

mkdir $modeldir

fi
cp finetune.sh $modeldir

python main.py $dataset\
               --benchmark $benchmark\
               --pretrained\
               -a $arch\
               -p 100\
               --epochs $epoch\
               --lr $lr\
               --lr-method $lr_method\
               --lr-params $lr_params\
               -j 8\
               -b $batchsize\
               --num-classes $num_classes\
               --representation $image_representation\
               --freezed-layer $freeze_layer\
               --classifier-factor $classifier_factor\
               --benchmark $benchmark\
               --modeldir $modeldir

else
checkpointfile=$(ls -rt $modeldir/*.pth.tar | tail -1)

python main.py $dataset\
               --benchmark $benchmark\
               --pretrained\
               -a $arch\
               -p 100\
               --epochs $epoch\
               --lr $lr\
               --lr-method $lr_method\
               --lr-params $lr_params\
               -j 8\
               -b $batchsize\
               --num-classes $num_classes\
               --representation $image_representation\
               --freezed-layer $freeze_layer\
               --modeldir $modeldir\
               --classifier-factor $classifier_factor\
               --benchmark $benchmark\
               --resume $checkpointfile

fi
echo "Done!"
