th main.lua -data gen/Flicker8k_Dataset -dataset flickr8k -finetuneAfter 100 -retrain pretrained/resnet-18.t7 -resetCNNlastlayer true -cnnType resnet

th main.lua -data gen/Flicker8k_Dataset -dataset flickr8k -finetuneAfter 100 -retrain pretrained/resnet-18.t7 -resetCNNlastlayer true -cnnType resnet -skipFlag true

th main.lua -data gen/Flicker8k_Dataset -dataset flickr8k -finetuneAfter 100 -cnnType vggnet -skipFlag true -cnnCaffe pretrained/VGG_ILSVRC_16_layers.caffemodel -cnnProto pretrained/VGG_ILSVRC_16_layers_deploy.prototxt -cnnCaffelayernum 38

th main.lua -data gen/Flicker8k_Dataset -dataset flickr8k -finetuneAfter 100 -cnnType vggnet -skipFlag true -cnnCaffe pretrained/VGG_ILSVRC_16_layers.caffemodel -cnnProto pretrained/VGG_ILSVRC_16_layers_deploy.prototxt -cnnCaffelayernum 38 -checkEvery -1 -nGPU 0 -backend nn

th main.lua -data gen/Flicker8k_Dataset -dataset flickr8k -finetuneAfter 100 -retrain pretrained/resnet-18.t7 -resetCNNlastlayer true -cnnType resnet -skipFlag true -checkEvery -1 -nThreads 2

