# NN-Learning

## Introduction

This is torch7 code for my nn learning.

Here are some greate code sources:

> [torch-rnn by jcjohnson](https://github.com/jcjohnson/torch-rnn)
>
> [torch-resnet by facebook](https://github.com/facebook/fb.resnet.torch)
>
> [torch-toolbox by e-lab](https://github.com/e-lab/torch-toolbox)
>
> [trainplot by joeyhng](https://github.com/joeyhng/trainplot)

## Mnist

Basic torch code for classifying MNIST dataset using LeNet. The code organization follows [torch-rnn by jcjohnson](https://github.com/jcjohnson/torch-rnn)

**NOTE**: checkpoint can save model but there are some problem continuing training from checkpoint because I forgot to save optimState.

## Cifar

Basic torch code for classifying Cifar10 dataset using VGGnet. The code organization follows [torch-rnn by jcjohnson](https://github.com/jcjohnson/torch-rnn)

**NOTE**: checkpoint can save model but there are some problem continuing training from checkpoint because I forgot to save optimState.

## Image

Basic torch code for different datasets using different model types. The code organization follows (some codes are copied from) [torch-resnet by facebook](https://github.com/facebook/fb.resnet.torch)

| Dataset		| Model									|
| ------------- |:-------------:						|
| mnist			| lenet									|
| cifar10		| lenet, vggnet, resnet, preresnet		|
| cifar100		| lenet, vggnet, resnet, preresnet		|
| (imagenet)	| lenet, vggnet, resnet, preresnet		|

**NOTE**:

	1. still working on imagenet, resnet, preresnet.

	2. dataloader is single thread version - working on multithread version

	3. please read the options when running