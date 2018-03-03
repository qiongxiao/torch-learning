# Torch - Cifar - VGGnet

## Introduction

Old version -- recommend using nn_learning/torch/Image

Basic Torch code for classifying cifar10 dataset. The accuracy rate can be up to 89% after 50 epochs.

## VGGnet

Here are some layer definitions:

1.
> define Conv [nInputPlane, nOutputPlane, kernelHeight, kernelWidth, hStride, wStride, hPadding, wPadding]

2.
> define ConvBNReLU[nInputPlane, nOutputPlane] as follows,
>
> Conv[nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1] --> SpatialBatchNorm[nOutputPlane] --> ReLU()

3.
> define SpatialMaxPooling [kernelHeight, kernelWidth, hStride, wStride, hPadding, wPadding]
>
> define MaxPooling as SpatialMaxPooling[2, 2, 2, 2, 0, 0]

VGGnet Construction shows as follows,

> ConvBNReLU[3, 64] --> Dropout --> ConvBNReLU[64, 64] --> MaxPooling -->
> ConvBNReLU[64, 128] --> Dropout --> ConvBNReLU[128, 128] --> MaxPooling -->
> ConvBNReLU[128, 256] --> Dropout --> ConvBNReLU[256, 256] --> Dropout --> ConvBNReLU[256, 256] --> MaxPooling -->
> ConvBNReLU[256, 512] --> Dropout --> ConvBNReLU[512, 512] --> Dropout --> ConvBNReLU[512, 512] --> MaxPooling -->
> ConvBNReLU[512, 512] --> Dropout --> ConvBNReLU[512, 512] --> Dropout --> ConvBNReLU[512, 512] --> MaxPooling -->
> View --> Dropout -->
> FC[512, 512] --> batchnorm --> ReLU --> Dropout
> FC[512, 10] - (-> using softmax)

Code style and organization follow [torch-rnn by jcjohnson](https://github.com/jcjohnson/torch-rnn). Therefore, some codes in train.lua are copied and edited from [torch-rnn by jcjohnson](https://github.com/jcjohnson/torch-rnn).

"weight-init.lua" is downloaded from [torch-toolbox by e-lab](https://github.com/e-lab/torch-toolbox)

"utils.lua" comes from [torch-rnn by jcjohnson](https://github.com/jcjohnson/torch-rnn) and [trainplot by joeyhng](https://github.com/joeyhng/trainplot)

## Usage

To train a model and use it to classify cifar10 data, you'll need to follow two simple steps:

### Step 1: Train the model

You can run the training script like this:

> th train.lua

You can change the batchsize and max_epoches like this

> th train.lua -batch_size 100 -max_epochs 20

You can observe training situation by open plot/showplot.html?path=out.json

### Step 2: Test the model

Run like this

>th test.lua -checkpoint cv/checkpoint_20000.t7

This will load the trained checkpoint cv/checkpoint_20000.t7 from the previous step, test the MNIST data from it.

**NOTE**: 

1. "BatchFlip.lua" comes from [cifar.torch by szagoruyko](https://github.com/szagoruyko/cifar.torch). I have NOT tried it yet and thus I do NOT whether there's bug.

2. cifar10 dataset is divided into 3 parts: train(45000), val(5000), test(10000)
