# Torch - NeuralTalk (LSTM)

## Introduction

Basic torch code for captioning images on flickr8k, mscoco dataset. Realiazation of [NeuralTalk2 by karpathy](https://github.com/karpathy/neuraltalk2) 

The code organization follows (some codes are copied from) [torch-resnet by facebook](https://github.com/facebook/fb.resnet.torch)

"plotter.lua" comes from [torch-rnn by jcjohnson](https://github.com/jcjohnson/torch-rnn) and [trainplot by joeyhng](https://github.com/joeyhng/trainplot)

## Usage

### Train the model and Test

You can run the training script using the command shown in comm.sh, such as,

> th main.lua -data gen/Flicker8k_Dataset -dataset flickr8k -finetuneAfter -1 -cnnType vggnet -skipFlag true -cnnCaffe pretrained/VGG_ILSVRC_16_layers.caffemodel -cnnProto pretrained/VGG_ILSVRC_16_layers_deploy.prototxt -cnnCaffelayernum 38 -checkEvery 1 -backendCaffe cudnn -maxCheckpointsNum 5

Or

> th main.lua -data gen/Flicker8k_Dataset -dataset flickr8k -finetuneAfter 100 -retrain pretrained/resnet-18.t7 -resetCNNlastlayer true -cnnType resnet

You can observe training situation by open plot/showplot.html?path=out.json

##NOTE:

1. please read the options (opts.lua) when running

2. multi-threaded and multi-GPU is unverified!!!!
