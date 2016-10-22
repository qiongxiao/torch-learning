# Torch - LeNet

## Introduction

Basic Torch code for classifying MNIST dataset. The accuracy rate can be up to 99.20% after 10 epochs.

LeNet Construction shows as follows,

> CONV[32, 5, 5] -> ReLU -> POOL[2, 2] -> CONV[64, 5, 5] -> ReLU -> POOL[2, 2] -> Dropout -> FC[,1024] -> ReLU -> FC[1024, 10] (-> using softmax)

Code style and organization follow [torch-rnn by jcjohnson](https://github.com/jcjohnson/torch-rnn). Therefore, some codes in train.lua are copied and edited from [torch-rnn by jcjohnson](https://github.com/jcjohnson/torch-rnn).

"weight-init.lua" is downloaded from [torch-toolbox by e-lab](https://github.com/e-lab/torch-toolbox)

"utils.lua" comes from [torch-rnn by jcjohnson](https://github.com/jcjohnson/torch-rnn) and [trainplot by joeyhng](https://github.com/joeyhng/trainplot)

## Usage

To train a model and use it to classify MNIST data, you'll need to follow two simple steps:

### Step 1: Train the model

You can run the training script like this:

> th train.lua

You can change the batchsize and max_epoches like this

> th train.lua -batch_size 100 -max_epoches 20

### Step 3: Test the model

Run like this

>th test.lua -checkpoint cv/checkpoint_20000.t7

This will load the trained checkpoint cv/checkpoint_20000.t7 from the previous step, test the MNIST data from it.