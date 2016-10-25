# Torch - CNN - models

## Introduction

Basic torch code for different datasets using different model types. 

| Dataset		| Model									|
| ------------- |:-------------:						|
| mnist			| lenet									|
| cifar10		| lenet, vggnet, resnet, preresnet		|
| cifar100		| lenet, vggnet, resnet, preresnet		|
| (imagenet)	| lenet, vggnet, resnet, preresnet		|

## Definition

> define iChanel : the number of chanels of input image
>
> define oSize : the height/width of output of conv stage before fully-connected layer
>
> define nClasses : the number of classes
>
> mnist: iChanel = 1, nClasses = 10
>
> cifar10 : iChanel = 3, nClasses = 10
>
> cifar100: iChanel = 3, nClasses = 100
>
> imagenet: iChanel = 3, nClasses = 1000

Here are some layer definitions:

> define Conv [nInputPlane, nOutputPlane, kernelHeight, kernelWidth, hStride, wStride, hPadding, wPadding]
>
> default Conv [nInputPlane, nOutputPlane, kernelHeight, kernelWidth, 1, 1, 0, 0]

> define SpatialMaxPooling [kernelHeight, kernelWidth, hStride, wStride, hPadding, wPadding]
>
> define maxPool as SpatialMaxPooling[2, 2, 2, 2, 0, 0]

> define define ConvBNReLU[nInputPlane, nOutputPlane]
>
> Conv[nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1] --> SpatialBatchNorm[nOutputPlane] --> ReLU()

## LeNet

> mnist : oSize = ((28 - 4) / 2 - 4) / 2 = 4
>
> cifar10/cifar100 :  oSize = ((32 - 4) / 2 - 4) / 2 = 5
>
> imagenet : ((224 - 4) / 2 - 4) / 2 = 53

LeNet Construction shows as follows,

> Conv[iChanel, 32, 5, 5] --> ReLU --> maxPool --> Conv[32, 64, 5, 5] -> ReLU -> maxPool -> Dropout -> FC[oSize*oSize*64, 1024] -> ReLU -> FC[1024, 10]

## VGGnet

> cifar10/cifar100 :  oSize = 32/(2^5) = 1
>
> imagenet : oSize = 224/(2^5) = 7

> define ConvBNReLU[nInputPlane, nOutputPlane] as follows,
>
> Conv[nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1] --> SpatialBatchNorm[nOutputPlane] --> ReLU()

VGGnet Construction shows as follows,

> ConvBNReLU[3, 64] --> Dropout --> ConvBNReLU[64, 64] --> MaxPooling -->
> ConvBNReLU[64, 128] --> Dropout --> ConvBNReLU[128, 128] --> MaxPooling -->
> ConvBNReLU[128, 256] --> Dropout --> ConvBNReLU[256, 256] --> Dropout --> ConvBNReLU[256, 256] --> MaxPooling -->
> ConvBNReLU[256, 512] --> Dropout --> ConvBNReLU[512, 512] --> Dropout --> ConvBNReLU[512, 512] --> MaxPooling -->
> ConvBNReLU[512, 512] --> Dropout --> ConvBNReLU[512, 512] --> Dropout --> ConvBNReLU[512, 512] --> MaxPooling -->
>
> View[512*oSize*oSize] --> Dropout --> FC[512, 512] -->
>
> (or Conv[512, 512, oSize, oSize] -->)
>
> batchnorm --> ReLU --> Dropout --> FC[512, 10] - (-> using softmax)