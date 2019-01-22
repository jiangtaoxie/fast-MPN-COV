set -e
:<<!
*****************Instruction*****************
For fine-tuning a model in two stage way.
For example, in the first stage, only the
classifier is tuned, the previous layers act like
a feature extactor, then all layers join the
train phase.
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
#mpncovresnet:mpncovresnet50, mpncovresnet101
#inceptionv3
#You can also add your own network in src/network
arch=vgg16
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
image_representation=BCNN
# short description of method
description=reproduce
#*********************************************

#*******************Dataset*******************
#Choose the dataset folder
benchmark=Dataset Name
datadir=/path/to/the/data
dataset=$datadir/$benchmark
num_classes=200
#*********************************************

#****************Hyper-parameters*************

# Freeze the layers before a certain layer.
freeze_layer=30
# Batch size
batchsize=40
# The number of total epochs for training
epoch=100
# The inital learning rate
# decreased by step method
lr=0.001
lr_method=step
lr_params=100
# log method
# description: lr = logspace(params1, params2, #epoch)

#lr_method=log
#lr_params=-1.1\ -5.0
#
weight_decay=5e-4
classifier_factor=1000
#*********************************************

#*************** First stage *****************
echo "Start finetuning the first satge!"
modeldir=Results/Finetune-$benchmark-$arch-$image_representation-$description-lr$lr-bs$batchsize
if [ ! -d  "Results" ]; then

mkdir Results

fi
if [ ! -d  "$modeldir" ]; then

mkdir $modeldir

fi

if [ ! -e $modeldir/*.pth.tar ]; then

cp two_stage_finetune.sh $modeldir

python main.py $dataset\
               --pretrained\
               -a $arch\
               -p 100\
               --epochs $epoch\
               --lr $lr\
               --lr-method $lr_method\
               --lr-params $lr_params\
               --wd $weight_decay\
               -j 4\
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
               --pretrained\
               -a $arch\
               -p 100\
               --epochs $epoch\
               --lr $lr\
               --lr-method $lr_method\
               --lr-params $lr_params\
               --wd $weight_decay\
               -j 4\
               -b $batchsize\
               --num-classes $num_classes\
               --representation $image_representation\
               --freezed-layer $freeze_layer\
               --modeldir $modeldir\
               --classifier-factor $classifier_factor\
               --benchmark $benchmark\
               --resume $checkpointfile

fi
#*********************************************

#*************** Second stage ****************
echo "Start finetuning the second satge!"
new_modeldir=$modeldir/second_stage

if [ ! -d  "$new_modeldir" ]; then

mkdir $new_modeldir

fi

cp $modeldir/stats.mat $new_modeldir/stats.mat


# Freeze the layers before a certain layer.
freeze_layer=0
# Batch size
batchsize=10
# The number of total epochs for training
# (including the first stage)
epoch=200
lr=0.001
lr_method=step
lr_params=200

if [ ! -e $new_modeldir/*.pth.tar ]; then

checkpointfile=$(ls -rt $modeldir/*.pth.tar | tail -1)
python main.py $dataset\
               --pretrained\
               -a $arch\
               -p 100\
               --epochs $epoch\
               --lr $lr\
               --lr-method $lr_method\
               --lr-params $lr_params\
               --wd $weight_decay\
               -j 4\
               -b $batchsize\
               --num-classes $num_classes\
               --representation $image_representation\
               --freezed-layer $freeze_layer\
               --classifier-factor $classifier_factor\
               --modeldir $new_modeldir\
               --benchmark $benchmark\
               --resume $checkpointfile

else

checkpointfile=$(ls -rt $new_modeldir/*.pth.tar | tail -1)

python main.py $dataset\
               --pretrained\
               -a $arch\
               -p 100\
               --epochs $epoch\
               --lr $lr\
               --lr-method $lr_method\
               --lr-params $lr_params\
               --wd $weight_decay\
               -j 4\
               -b $batchsize\
               --num-classes $num_classes\
               --representation $image_representation\
               --freezed-layer $freeze_layer\
               --modeldir $new_modeldir\
               --classifier-factor $classifier_factor\
               --benchmark $benchmark\
               --resume $checkpointfile

fi
echo "Done!"
#*********************************************
