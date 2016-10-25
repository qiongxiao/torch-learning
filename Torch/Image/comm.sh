th main.lua -dataset cifar10 -nGPU 1 -gen gen -dataAug 0 -colorspace yuv -maxEpochs 120 -startEpoch 1 -batchsize 128 -tenCrop false -save plot/out -plotEvery 0 -optimizer sgd -lr 1 -lr_decay 1e-7 -weight_decay 5e-4 -momentum 0.9 -decay_every 25 -decay_factor 0.5 -netType VGG -nClasses 10

th main.lua -dataset cifar10 -nGPU 1 -gen gen -dataAug 1 -colorspace rgb -maxEpochs 120 -startEpoch 1 -batchsize 128 -tenCrop false -save plot/out -plotEvery 0 -optimizer sgd -lr 1 -lr_decay 1e-7 -weight_decay 5e-4 -momentum 0.9 -decay_every 25 -decay_factor 0.5 -netType VGG -nClasses 10

th main.lua -dataset cifar10 -nGPU 1 -gen gen -dataAug 0 -colorspace rgb -maxEpochs 120 -startEpoch 1 -batchsize 128 -tenCrop false -save plot/out -plotEvery 0 -optimizer adam -lr 1e-2 -lr_decay 0 -weight_decay 0 -momentum 0 -decay_every 5 -decay_factor 0.95 -netType VGG -nClasses 10

th main.lua -dataset MNIST -nGPU 1 -gen gen -dataAug 0 -colorspace rgb -maxEpochs 100 -startEpoch 1 -batchsize 128 -tenCrop false -save plot/out -plotEvery 0 -optimizer adam -lr 1e-2 -lr_decay 0 -weight_decay 0 -momentum 0 -decay_every 10 -decay_factor 0.9 -netType LeNet -nClasses 10