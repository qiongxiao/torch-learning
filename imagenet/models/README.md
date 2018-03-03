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

> define iChannels : the number of chanels of input image
>
> define oSize : the height/width of output of conv stage before fully-connected layer
>
> define nClasses : the number of classes
>
> mnist: iChannels = 1, nClasses = 10
>
> cifar10 : iChannels = 3, nClasses = 10
>
> cifar100: iChannels = 3, nClasses = 100
>
> imagenet: iChannels = 3, nClasses = 1000

Here are some layer definitions:

1. 
> define Conv [nInputPlane, nOutputPlane, kernelHeight, kernelWidth, hStride, wStride, hPadding, wPadding]
>
> default Conv [nInputPlane, nOutputPlane, kernelHeight, kernelWidth, 1, 1, 0, 0]

2. 
> define SpatialMaxPooling [kernelHeight, kernelWidth, hStride, wStride, hPadding, wPadding]
>
> define maxPool as SpatialMaxPooling[2, 2, 2, 2, 0, 0]

3. 
> define ConvBNReLU[nInputPlane, nOutputPlane]
>
> Conv[nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1] --> SpatialBatchNorm[nOutputPlane] --> ReLU()

## LeNet

> mnist : oSize = ((28 - 4) / 2 - 4) / 2 = 4
>
> cifar10/cifar100 :  oSize = ((32 - 4) / 2 - 4) / 2 = 5
>
> imagenet : ((224 - 4) / 2 - 4) / 2 = 53

LeNet Construction shows as follows,

> Conv[iChannels, 32, 5, 5] --> ReLU --> maxPool --> Conv[32, 64, 5, 5] -> ReLU -> maxPool -> Dropout -> FC[oSize*oSize*64, 1024] -> ReLU -> FC[1024, 10]

## VGGnet

> cifar10/cifar100 :  oSize = 32/(2^5) = 1
>
> imagenet : oSize = 224/(2^5) = 7

new layer definition:
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
> View[512*oSize*oSize] --> Dropout --> FC[512*oSize*oSize, 4096] -->
>
> (or Conv[512, 512, oSize, oSize] -->)
>
> batchnorm --> ReLU --> Dropout --> FC[4096, nClasses] - (-> using softmax)

### ResNet

new layer definition:

1.
> shortcutA[nInputPlane, nOutputPlane, stride]
>
> if nOutputPlane == nInputPlane
>
> nn.Identity()
>
> else
>
> nn.SpatialAveragePooling[1, 1, stride, stride] --> Concat(nn.Identity(), nn.MulConstant(0))

2.
> shortcutB[nInputPlane, nOutputPlane, stride]
>
> if nOutputPlane == nInputPlane
>
> nn.Identity()
>
> else
>
> Conv[nInputPlane, nOutputPlane, 1, 1, stride, stride] --> SpatialBatchNorm[nOutputPlane]

3.
> define basicblock[nOutputPlane, stride]
>
> Conv[nInputPlane, nOutputPlane, 3, 3, stride, stride, 1, 1] --> SpatialBatchNorm[nOutputPlane] --> ReLU() --> Conv[nOutputPlane, nOutputPlane, 3, 3, stride, stride, 1, 1] -->  SpatialBatchNorm[nOutputPlane] {--> Add(shortcut_type[nInputPlane, nOutputPlane, stride])} --> ReLU()

4.
> define bottleneck[nOutputPlane, stride]
>
> Conv[nInputPlane, nOutputPlane, 1, 1, stride, stride, 1, 1] --> SpatialBatchNorm[nOutputPlane] --> ReLU() --> Conv[nOutputPlane, nOutputPlane, 3, 3, stride, stride, 1, 1] --> SpatialBatchNorm[nOutputPlane] --> ReLU() --> Conv[nOutputPlane, nOutputPlane * 4, 3, 3, stride, stride, 1, 1] --> SpatialBatchNorm[nOutputPlane * 4] --> {--> Add[nInputPlane, nOutputPlane * 4, stride])} --> ReLU()

#### cifar10

default shortcut type = A

> define n = (depth - 2) / 6 - 1 (n must be integer)
>
> ConvBNReLU[3, 16] --> { basicblock[16, 1] }\*(n+1) --> basicblock[32, 2] --> { basicblock[32, 1] }\*n --> basicblock[64, 2] --> { basicblock[64, 1] }\*n -->  SpatialAveragePooling[8, 8, 1, 1] -->
>
> View --> FC[64, 10]

size changes
> 3\*32\*32 --> 16\*32\*32 --> 16\*32\*32 --> 32\*16\*16 --> 32\*16\*16 --> 64\*8\*8 --> 64\*8\*8 --> 64\*1\*1

#### cifar100

> same as cifar10
>
> View --> FC[64, 100]

#### imagenet

See the code

Take depth = 34 as example.

> Conv[3, 64, 7, 7, 2, 2, 3, 3] --> SpatialBatchNorm() -> ReLU() -> SpatialMaxPooling[3, 3, 2, 2, 1, 1] --> { bottleneck[64, 1] }\*3 --> bottleneck[128, 2] --> { bottleneck[128, 1] }\*3 --> bottleneck[256, 2] --> { bottleneck[256, 1] }\*5 --> bottleneck[512, 2] --> { bottleneck[512, 1] }\*2 --> SpatialAveragePooling[7, 7, 1, 1] -->
>
> View --> FC[512, 1000]

size changes
> 3\*224\*224 --> 64\*112\*112 --> 64\*56\*56 --> 64\*56\*56 --> 128\*28\*28 --> 128\*28\*28 --> 256\*14\*14 --> 256\*14\*14 --> 512\*7\*7 --> 512\*7\*7 --> 512\*1\*1
