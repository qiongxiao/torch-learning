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
	cmd:option('-data',			'',			'Path to dataset')
	cmd:option('-dataset',		'mscoco',	'Options: flickr8k | mscoco')
	cmd:option('-manualSeed',	123,		'Manually set RNG seed')
	cmd:option('-backend',		'cudnn',	'Option: cudnn | nn')
	cmd:option('-verbose',		'true',		'whether verbose a example prediction caption for every val batch: false | true')
	cmd:option('-nGPU',			1,			'GPU number: 0 CPU | positve integer')
	------------- Data options ------------------------
	cmd:option('-dataAug',				0,		'whether augment data : 0 (false) | 1 (true)')
	cmd:option('-nThreads',				1,		'number of data loading threads, positve integer')
	cmd:option('-seqLength',			30,		'length of caption, positve integer')
	cmd:option('-seqPerImg',			5,		'number of captions per image, positve integer')
	cmd:option('-wordCountThreshold',	5,		'threshould of the number of word appearance, positve integer')
	------------- Training options --------------------
	cmd:option('-maxEpochs',		0,			'Number of total epochs to run')
	cmd:option('-batchsize',		16,			'mini-batch size (1 = pure stochastic)')
	cmd:option('-testOnly',			'false',	'Run on validation set only')
	cmd:option('-finetuneAfter',	-1,			'finetune after * epochs: -1 disable | positve integer')
	cmd:option('-startEpoch',		1,			'staring epoch number')
	------------- Checkpoint options ------------------
	cmd:option('-save',				'checkpoints',	'Directory in which to save checkpoints')
	cmd:option('-resume',			'none',			'Resume from the latest checkpoint in this directory')
	cmd:option('-resumeType',		'latest',		'Options: latest | best')
	cmd:option('-checkEvery',		1,				'checkpoint every # epcoh: -1 disable | positve integer')
	cmd:option('-maxCheckpointsNum',5,				'max number of checkpoints: -1 disable | positve interger')
	------------- Plotting options --------------------
	cmd:option('-plotPath',			'plot/out',	'Path to output plot file (excluding .json)')
	cmd:option('-plotEvery',		0,			'Whether to plot every iteration')
	------------- Optimization options ----------------
	cmd:option('-gradClip',			0.1,		'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
	cmd:option('-decay',			'default',	'using external decay parameter on lstm')
	cmd:option('-decay_every',		100,		'external learning rate decay')
	cmd:option('-decay_factor',		0.5,		'external learning rate decay factor')
	------------- lstm Optimization options -----------
	cmd:option('-optimizer',		'adam',		'lstm optimizer algorithm: adam | sgd')
	cmd:option('-lr',				4e-4,		'lstm initial learning rate')
	cmd:option('-lr_decay',			0,			'lstm learning rate decay')
	cmd:option('-weightDecay',		0,			'lstm weight decay')
	cmd:option('-momentum',			0.9,		'lstm momentum, for sgd')
	cmd:option('-optimAlpha',		0.8,		'lstm alpha for adam')
	cmd:option('-optimBeta',		0.999,		'lstm beta used for adam')
	------------- cnn Optimization options ------------
	cmd:option('-cnnOptimizer',		'adam',		'cnn optimizer algorithm: adam | sgd')
	cmd:option('-cnnLr',			1e-5,		'cnn initial learning rate')
	cmd:option('-cnnLr_decay',		0,			'cnn learning rate decay')
	cmd:option('-cnnWeigthDecay',	0,			'cnn weight decay')
	cmd:option('-cnnMomentum',		0.9,		'cnn momentum, for sgd')
	cmd:option('-cnnOptimAlpha',	0.8,		'cnn alpha for momentum of CNN')
	cmd:option('-cnnOptimBeta',		0.999,		'cnn beta for momentum of CNN')
	------------- cnn Model options -------------------
	cmd:option('-cnnType',			'vggnet',   'Options: resnet | vggnet')
	cmd:option('-cnnFCdropout',		0.5,		'From feature layer to encoding layer')
	------------- lstm Model options ------------------
	cmd:option('-skipFlag',			'false',	'whether to skip when input seq is over')
	cmd:option('-rDepth',			1,			'depth of lstm')
	cmd:option('-encodingSize',		512,		'size of encoding of lstm input')
	cmd:option('-hiddenStateSize',	512,		'size of hidden state in lstm')
	cmd:option('-lstmDropout',		0.5,		'dropout for  lstm')
	cmd:option('-inferenceMax',		1,			'using argmax algorithm to get word')
	cmd:option('-temperature',		1,			'using normal sample algorithm to get word')
	------------- cnn Model init options --------------
	cmd:option('-retrain',			'none',		'Path to cnn model (t7) to retrain with')
	cmd:option('-resetCNNlastlayer','false',	'Delete final FC layer of cnn, must provide cnnFeatures at the same time')
	cmd:option('-cnnCaffe',			'none',		'Path to caffe cnn model to retrain with')
	cmd:option('-cnnProto',			'none',		'Path to caffe cnn model prototxt')
	cmd:option('-cnnCaffelayernum',	38,			'the layer number of last feature layer in caffe cnn model')
	cmd:option('-backendCaffe',		'nn',		'Options: cudnn | nn (for caffe load)')
	cmd:option('-cnnFeatures',		4096,		'the feature number of last feature layer in cnn model')
	------------- lstm Model init options -------------
	cmd:option('-retrainlstm',		'none',		'Path to lstm model (t7) to retrain with')

	cmd:text()

	local opt = cmd:parse(arg or {})

	opt.verbose = opt.verbose ~= 'false'
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

	if opt.backend ~= 'cudnn' and opt.backend~= 'nn' then
		cmd:error('error: invalid backend '.. opt.backend)
	end

	if opt.finetuneAfter == 0 then
		cmd:error('error: invalid finetuneAfter value')
	end

	if opt.nGPU < 1 and (opt.backend == 'cudnn' or opt.backendCaffe == 'cudnn') then
		cmd:error('error: CPU mode has no cudnn backend')
	end

	return opt
end

return M
