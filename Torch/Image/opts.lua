--[[ code from https://github.com/facebook/fb.resnet.torch/blob/master/opts.lua
--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--]]

local M = { }


function M.parse(arg)
	local cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Torch-7 CNNet Training script')
	cmd:text('See for examples')
	cmd:text()
	cmd:text('Options:')
	------------- General options ---------------------
	cmd:option('-data',         '',         'Path to dataset')
	cmd:option('-dataset',      'cifar10',  'Options: imagenet | cifar10 | cifar100 | MNIST')
	cmd:option('-manualSeed',   0,          'Manually set RNG seed')
	cmd:option('-nGPU',         1,          'Number of GPUs to use by default')
	------------- Data options ------------------------
	cmd:option('-dataAug',      0,          'whether augment data')
	cmd:option('-colorspace',   'rgb',      'colorspace where normalization excutes')
	cmd:option('-nThreads',     1,          'number of data loading threads, positve integer')
	------------- Training options --------------------
	cmd:option('-maxEpochs',    0,          'Number of total epochs to run')
	cmd:option('-batchsize',    32,         'mini-batch size (1 = pure stochastic)')
	cmd:option('-testOnly',     'false',    'Run on validation set only')
	cmd:option('-tenCrop',      'false',    'Ten-crop testing')
	------------- Checkpoint options ------------------
	cmd:option('-save',         'checkpoints',  'Directory in which to save checkpoints')
	cmd:option('-resume',       'none',         'Resume from the latest checkpoint in this directory')
	cmd:option('-checkEvery',   1,              'checkpoint every epcoh #')
	------------- Plotting options --------------------
	cmd:option('-plotPath',     'plot/out',     'Path to output plot file (excluding .json)')
	cmd:option('-plotEvery',    0,              'Whether to plot every iteration')
	------------- Optimization options ----------------
	cmd:option('-optimizer',        'adam',     'optimizer algorithm: adam | sgd')
	cmd:option('-lr',               0.1,        'initial learning rate')
	cmd:option('-lr_decay',         0,          'learning rate decay')
	cmd:option('-weightDecay',      1e-4,       'weight decay')
	cmd:option('-momentum',         0.9,        'momentum, for sgd')
	cmd:option('-decay',            'default',  'whether using external decay parameter')
	cmd:option('-decay_every',      25,         'external learning rate decay')
	cmd:option('-decay_factor',     0.5,        'external learning rate decay factor')
	cmd:option('-dropout',          0.5,        'dropout for fully-connected layer')
	cmd:option('-convDropout',      0.5,        'dropout for convolution layer')
	------------- Model options -----------------------
	cmd:option('-netType',          'resnet',   'Options: resnet | preresnet | vggnet | lenet | SmallCNNet')
	cmd:option('-depth',            34,         'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
	cmd:option('-shortcutType',     '',         'Options: A | B | C')
	cmd:option('-retrain',          'none',     'Path to model to retrain with')
	------------- Model options -----------------------
	cmd:option('-shareGradInput',   'false',    'Share gradInput tensors to reduce memory usage')
	cmd:option('-optnet',           'false',    'Use optnet to reduce memory usage')
	cmd:option('-resetClassifier',  'false',    'Reset the fully connected layer for fine-tuning')
	cmd:option('-nClasses',         0,          'Number of classes in the dataset')
	cmd:text()

	local opt = cmd:parse(arg or {})

	opt.testOnly = opt.testOnly ~= 'false'
	opt.tenCrop = opt.tenCrop ~= 'false'
	opt.shareGradInput = opt.shareGradInput ~= 'false'
	opt.optnet = opt.optnet ~= 'false'
	opt.resetClassifier = opt.resetClassifier ~= 'false'

	if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
		cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
	end

	if opt.dataset == 'imagenet' then
		-- Handle the most common case of missing -data flag
		local trainDir = paths.concat(opt.data, 'train')
		if not paths.dirp(opt.data) then
			cmd:error('error: missing ImageNet data directory')
		elseif not paths.dirp(trainDir) then
			cmd:error('error: ImageNet missing `train` directory: ' .. trainDir)
		end
		-- Default shortcutType=B and nEpochs=90
		opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
		opt.maxEpochs = opt.maxEpochs == 0 and 90 or opt.maxEpochs
	elseif opt.dataset == 'cifar10' then
		-- Default shortcutType=A and nEpochs=164
		opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
		opt.maxEpochs = opt.maxEpochs == 0 and 164 or opt.maxEpochs
	elseif opt.dataset == 'cifar100' then
		-- Default shortcutType=A and nEpochs=164
		opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
		opt.maxEpochs = opt.maxEpochs == 0 and 164 or opt.maxEpochs
	elseif opt.dataset == 'mnist' then
		opt.maxEpochs = opt.maxEpochs == 0 and 100 or opt.maxEpochs
	else
		cmd:error('unknown dataset: ' .. opt.dataset)
	end

	if opt.resetClassifier then
		if opt.nClasses == 0 then
			cmd:error('-nClasses required when resetClassifier is set')
		end
	end

	if opt.shareGradInput and opt.optnet then
		cmd:error('error: cannot use both -shareGradInput and -optnet')
	end

	if opt.nThreads < 1 then
		cmd:error('error: invalid threads number')
	end

	return opt
end

return M
