th main.lua -dataset cifar10 -nGPU 1 -dataAug 0 -colorspace yuv -maxEpochs 120 -batchsize 128 -tenCrop false -checkEvery 1 -plotPath plot/out -plotEvery 0 -optimizer sgd -lr 1 -lr_decay 1e-7 -weightDecay 5e-4 -momentum 0.9 -decay 0 -decay_every 25 -decay_factor 0.5 -dropout 0.5 -convDropout 0.4 -netType vggnet

th main.lua -dataset cifar10 -nGPU 1 -dataAug 1 -colorspace rgb -maxEpochs 120 -batchsize 128 -tenCrop false -checkEvery 1 -plotPath plot/out -plotEvery 0 -optimizer sgd -lr 1 -lr_decay 1e-7 -weightDecay 5e-4 -momentum 0.9 -decay 0 -decay_every 25 -decay_factor 0.5 -dropout 0.5 -convDropout 0.4 -netType vggnet

th main.lua -dataset cifar10 -nGPU 1 -dataAug 0 -colorspace rgb -maxEpochs 120 -batchsize 128 -tenCrop false -checkEvery 1 -plotPath plot/out -plotEvery 0 -optimizer adam -lr 1e-2 -lr_decay 0 -weightDecay 0 -momentum 0 -decay 0 -decay_every 5 -decay_factor 0.95 -dropout 0.5 -convDropout 0.4 -netType vggnet

th main.lua -dataset mnist -nGPU 1 -dataAug 0 -colorspace rgb -maxEpochs 100 -batchsize 128 -tenCrop false -checkEvery 0 -plotPath plot/out -plotEvery 0 -optimizer adam -lr 1e-2 -lr_decay 0 -weightDecay 0 -momentum 0 -decay 0 -decay_every 10 -decay_factor 0.9 -netType lenet
