--[[
--
-- code from https://github.com/facebook/fb.resnet.torch/blob/master/opts.lua
--
--]]

local M = { }


function M.parse(arg)
	local cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Torch-7 Image Captioning Training script')
	cmd:text('See for examples')
	cmd:text()
	cmd:text('Options:')
	------------- General options ---------------------
	cmd:option('-data',			'',				'Path to dataset')
	cmd:option('-dataset',		'mscoco',		'Options: flickr8k | mscoco')
	cmd:option('-manualSeed',	123,			'Manually set RNG seed')
	------------- Data options ------------------------
	cmd:option('-dataAug',				0,		'whether augment data')
	cmd:option('-nThreads',				1,		'number of data loading threads, positve integer')
	cmd:option('-seqLength',			30,		'length of caption, positve integer')
	cmd:option('-seqPerImg',			5,		'number of captions per image, positve integer')
	cmd:option('-wordCountThreshold',	5,		'threshould of the number of word appearance, positve integer')
	------------- Training options --------------------
	cmd:option('-maxEpochs',		0,			'Number of total epochs to run')
	cmd:option('-batchsize',		16,			'mini-batch size (1 = pure stochastic)')
	cmd:option('-testOnly',			'false',	'Run on validation set only')
	cmd:option('-finetune',			0,			'whether finetuning')
	cmd:option('-finetuneAfter',	50,			'finetune after * epochs')
	------------- Checkpoint options ------------------
	cmd:option('-save',			'checkpoints',	'Directory in which to save checkpoints')
	cmd:option('-resume',		'none',			'Resume from the latest checkpoint in this directory')
	cmd:option('-checkEvery',	1,				'checkpoint every epcoh #')
	------------- Plotting options --------------------
	cmd:option('-plotPath',			'plot/out',	'Path to output plot file (excluding .json)')
	cmd:option('-plotEvery',		0,			'Whether to plot every iteration')
	------------- Optimization options ----------------
	cmd:option('-gradClip',			0.1,		'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
	cmd:option('-optimizer',		'adam',		'lstm optimizer algorithm: adam | sgd')
	cmd:option('-lr',				4e-4,		'lstm initial learning rate')
	cmd:option('-lr_decay',			0,			'lstm learning rate decay')
	cmd:option('-weightDecay',		1e-4,		'lstm weight decay')
	cmd:option('-momentum',			0.9,		'lstm momentum, for sgd')
	cmd:option('-optimAlpha',		0.8,		'alpha for adam')
	cmd:option('-optimBeta',		0.999,		'beta used for adam')
	cmd:option('-decay',			'default',	'lstm using external decay parameter on lstm')
	cmd:option('-decay_every',		100,		'lstm external learning rate decay')
	cmd:option('-decay_factor',		0.5,		'lstm external learning rate decay factor')
	------------- cnn Optimization options ----------------
	cmd:option('-cnnOptimizer',		'adam',		'cnn optimizer algorithm: adam | sgd')
	cmd:option('-cnnLr',			1e-5,		'initial cnn learning rate')
	cmd:option('-cnnLr_decay',		0,			'cnn learning rate decay')
	cmd:option('-cnnWeigthDecay',	0,			'cnn weight decay')
	cmd:option('-cnnMomentum',		0.9,		'cnn momentum, for sgd')
	cmd:option('-cnnOptimAlpha',	0.8,		'alpha for momentum of CNN')
	cmd:option('-cnnOptimBeta',		0.999,		'beta for momentum of CNN')
	cmd:option('-cnnDecay',			'default',	'cnn using external decay parameter on cnn')
	cmd:option('-cnnDecay_every',	25,			'external learning rate decay')
	cmd:option('-cnnDecay_factor',	0.5,		'external learning rate decay factor')	
	------------- cnn Model options -----------------------
	cmd:option('-cnnType',			'vggnet',   'Options: resnet | vggnet')
	cmd:option('-cnnFCdropout',		0.5,		'dropout for cnn fully-connected layer')
	cmd:option('-cnnCONVdropout',	0.5,		'dropout for cnn convolution layer')
	cmd:option('-resnetDepth',		34,			'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
	cmd:option('-shortcutType',		'',			'ResNet Options: A | B')
	------------- lstm Model options -----------------------
	cmd:option('-skipFlag',			'false',	'whether to skip after input seq is over')
	cmd:option('-rDepth',			1,			'depth of lstm')
	cmd:option('-encodingSize',		512,		'size of encoding of lstm input')
	cmd:option('-hiddenStateSize',	512,		'size of hidden state in lstm')
	cmd:option('-lstmDropout',		0.5,		'dropout for  lstm')
	cmd:option('-inferenceMax',		1,			'using argmax algorithm to get word')
	cmd:option('-temperature',		1,			'using normal sample algorithm to get word')
	------------- Model options -----------------------
	cmd:option('-retrain',			'none',		'Path to cnn model to retrain with')
	cmd:option('-resetCNNlastlayer','false',	'Reset the fully connected layer for fine-tuning')

	cmd:text()

	local opt = cmd:parse(arg or {})

	opt.testOnly = opt.testOnly ~= 'false'
	opt.resetCNNlastlayer = opt.resetCNNlastlayer ~= 'false'
	opt.skipFlag = opt.skipFlag ~= 'false'

	if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
		cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
	end

	if opt.dataset == 'mscoco' then
		-- Handle the most common case of missing -data flag
		local trainDir = paths.concat(opt.data, 'train2014')
		if not paths.dirp(opt.data) then
			cmd:error('error: missing ImageNet data directory')
		elseif not paths.dirp(trainDir) then
			cmd:error('error: ImageNet missing `train2014` directory: ' .. trainDir)
		end
		-- Default shortcutType=B and nEpochs=90
		opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
		opt.maxEpochs = opt.maxEpochs == 0 and 90 or opt.maxEpochs
	elseif opt.dataset == 'flickr8k' then
		opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
		opt.maxEpochs = opt.maxEpochs == 0 and 90 or opt.maxEpochs
	else
		cmd:error('unknown dataset: ' .. opt.dataset)
	end

	if opt.nThreads < 1 then
		cmd:error('error: invalid threads number')
	end

	return opt
end

return M
