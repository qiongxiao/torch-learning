# Torch - CNN

## Introduction

Basic torch code for different datasets using different model types. 

| Dataset		| Model						|
| ------------- |:-------------:			|
| mnist			| lenet						|
| cifar10		| lenet, vggnet, resnet		|
| cifar100		| lenet, vggnet, resnet		|
| imagenet		| lenet, vggnet, resnet		|

The code organization follows (some codes are copied from) [torch-resnet by facebook](https://github.com/facebook/fb.resnet.torch)

"plotter.lua" comes from [torch-rnn by jcjohnson](https://github.com/jcjohnson/torch-rnn) and [trainplot by joeyhng](https://github.com/joeyhng/trainplot)

## Usage

### Train the model and Test

You can run the training script using the command shown in comm.sh, such as,

> th main.lua -dataset mnist -nGPU 1 -dataAug 0 -colorspace rgb -maxEpochs 50 -startEpoch 1 -batchsize 128 -tenCrop false -checkEvery 1 -plotPath plot/out -plotEvery 0 -optimizer adam -lr 1e-2 -lr_decay 0 -weightDecay 0 -momentum 0 -decay 0 -decay_every 10 -decay_factor 0.9 -netType lenet

Or

> th main.lua -dataset cifar10 -nGPU 1 -dataAug 0 -colorspace rgb -maxEpochs 165 -startEpoch 1 -batchsize 128 -tenCrop false -checkEvery 1 -plotPath plot/out -plotEvery 0 -optimizer adam -lr 1e-2 -lr_decay 0 -weightDecay 0 -momentum 0 -decay 0 -decay_every 5 -decay_factor 0.95 -dropout 0.5 -convDropout 0.4  -netType vggnet

Or

> th main.lua -dataset cifar10 -nGPU 1 -dataAug 0 -colorspace yuv -maxEpochs 165 -startEpoch 1 -batchsize 128 -tenCrop false -checkEvery 1 -plotPath plot/out -plotEvery 0 -optimizer sgd -lr 1 -lr_decay 1e-7 -weightDecay 5e-4 -momentum 0.9 -decay 0 -decay_every 25 -decay_factor 0.5 -dropout 0.5 -convDropout 0.4 -netType vggnet

You can observe training situation by open plot/showplot.html?path=out.json

### Only Test the model

Run like this

>th main.lua -dataset mnist -nGPU 1 -dataAug 0 -colorspace rgb -testOnly true -batchsize 128 -tenCrop false -netType lenet -resume checkpoints/model_best.t7

This will load the trained checkpoint checkpoints/model_best.t7 from the previous step, test the mnist data from it.

##NOTE:

1. please read the options (opts.lua) when running

2. multi-threaded version is unverified!!!!
