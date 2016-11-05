th main.lua -data gen/Flicker8k_Dataset -dataset flickr8k -finetune 1 -finetuneAfter 50 -retrain pretrained/resnet-18.t7 -resetCNNlastlayer true -cnnType resnet

th main.lua -data gen/Flicker8k_Dataset -dataset flickr8k -finetune 1 -finetuneAfter 50 -retrain pretrained/resnet-18.t7 -resetCNNlastlayer true -cnnType resnet -skipFlag true
