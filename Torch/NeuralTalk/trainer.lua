--[[
--
--  code from https://github.com/facebook/fb.resnet.torch/blob/master/train.lua
--
--  The training loop and learning rate schedule
--
--]]
require 'torch'
require 'cutorch'
require 'nn'
local optim = require 'optim'

local M = {}
local Trainer = torch.class('cnn.Trainer', M)

function Trainer:__init(model, criterion, opt, optimConfig, cnnOptimConfig)
	self.cnn, self.expander, self.feature2seq = unpack(model)
	self.criterion = criterion

	self.optimizer = opt.optimizer
	if self.optimizer == 'sgd' then
		self.optimConfig = optimConfig or {
			learningRate = opt.lr,
			learningRateDecay = opt.lr_decay,
			weigthDecay = opt.weigthDecay,
			momentum = opt.momentum,
			nesterov = true,
			dampening = 0.0
		}
	elseif self.optimizer == 'adam' then
		self.optimConfig = optimConfig or {
			learningRate = opt.lr,
			learningRateDecay = opt.lr_decay,
			weigthDecay = opt.weigthDecay
		}
	else
		error('invalid optimizer ' .. opt.optimizer)
	end

	self.cnnOptimizer = opt.cnnOptimizer
	if self.cnnOptimizer == 'sgd' then
		self.cnnOptimConfig = cnnOptimConfig or {
			learningRate = opt.cnnLr,
			learningRateDecay = opt.cnnLr_decay,
			weigthDecay = opt.cnnWeigthDecay,
			momentum = opt.cnnMomentum,
			nesterov = true,
			dampening = 0.0
		}
	elseif self.cnnOptimizer == 'adam' then
		self.cnnOptimConfig = cnnOptimConfig or {
			learningRate = opt.cnnLr,
			learningRateDecay = opt.cnnLr_decay,
			weigthDecay = opt.cnnWeigthDecay
		}
	else
		error('invalid cnnOptimizer ' .. opt.cnnOptimizer)
	end

	self.opt = opt
	self.params, self.gradParams = self.feature2seq:getParameters()
	self.cnnParams, self.cnnGradParams = self.cnn:getParameters()
end

function Trainer:train(epoch, dataloader, finetune, plotter)
	-- Trains the model for a single epoch
	self.optimConfig.learningRate = self:learningRate(epoch)

	local timer = torch.Timer()
	local dataTimer = torch.Timer()

	local function cnnFeval()
		return self.cnn.output, self.cnnGradParams
	end
	local function feval()
		return self.criterion.output, self.gradParams
	end

	local trainSize = dataloader:size()
	local lossSum = 0.0
	local N = 0

	print('<Training> => Training epoch # ' .. epoch)
	-- set the batch norm to training mode
	self.model:training()
	for n, sample in dataloader:run() do
		local dataTime = dataTimer:time().real

		local batchsize = sample.input:size(1) / nCrops
		-- Copy input and target to the GPU
		self:copyInputs(sample)

		self.cnn:forward(self.input)
		self.expander:forward(self.cnn.output)
		self.feature2seq:forward({self.expander.output, self.target})
		local loss = self.criterion:forward(self.feature2seq.output, self.target)

		self.feature2seq:zeroGradParameters()
		if finetune == 1 then
			self.cnn:zeroGradParameters()
		end
		self.criterion:backward(self.feature2seq.output, self.target)
		self.feature2seq:backward({self.expander.output, self.target}, self.criterion.gradInput)

		if finetune == 1 then
			self.expander:backward(self.cnn.output, self.feature2seq.gradInput[1])
			self.cnn:backward(self.input, self.expander:gradInput)
		end

		if self.optimizer == 'sgd' then
			optim.sgd(feval, self.params, self.optimConfig)
		else
			optim.adam(feval, self.params, self.optimConfig)
		end

		if finetune == 1 then
			if self.cnnOptimizer == 'sgd' then
				optim.sgd(cnnFeval, self.cnnParams, self.cnnOptimConfig)
			else
				optim.adam(cnnFeval, self.cnnParams, self.cnnOptimConfig)
			end
		end

		lossSum = lossSum + loss*batchsize
		N = N + batchsize

		print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f'):format(
			epoch, n, trainSize, timer:time().real, dataTime, loss))

		-- check that the storage didn't get changed do to an unfortunate getParameters call
		assert(self.params:storage() == self.feature2seq:parameters()[1]:storage())
		assert(self.cnnParams:storage() == self.cnn:parameters()[1]:storage())

		timer:reset()
		dataTimer:reset()
		if self.opt.plotEvery == 1 then
			plotter:add('Train Loss - Iteration', 'Train', (epoch-1)*trainSize+n, loss)
		end
	end
	return lossSum / N
end

function Trainer:test(epoch, dataloader)

	local timer = torch.Timer()
	local dataTimer = torch.Timer()
	local size = dataloader:size()

	local nCrops = self.opt.tenCrop and 10 or 1
	local lossSum = 0.0
	local N = 0

	self.model:evaluate()
	for n, sample in dataloader:run() do
		local dataTime = dataTimer:time().real

		local batchsize = sample.input:size(1) / nCrops
		-- Copy input and target to the GPU
		self:copyInputs(sample)

		self.cnn:forward(self.input)
		self.expander:forward(self.cnn.output)
		self.feature2seq:forward({self.expander.output, self.target})
		local loss = self.criterion:forward(self.feature2seq.output, self.target)

		lossSum = lossSum + loss*batchsize
		N = N + batchsize

		local seq = self.feature2seq:inference(sefl.cnn.output)
		local out = dataloader:decode(seq)

		print((' | eval: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f'):format(
			epoch, n, size, timer:time().real, dataTime, loss))

		timer:reset()
		dataTimer:reset()
	end
	self.model:training()

	print((' * Finished epoch # %d    Err: %7.3f\n'):format(
		epoch, lossSum / N))

	return lossSum / N, out
end

function Trainer:inference(imgs, dataloader)

	self.model:evaluate()

	imgs:cuda()

	self.cnn:forward(imgs)
	local output = self.feature2seq:inference(self.cnn.output)
	local out = dataloader:decode(seq)

	self.model:training()
	return out
end

function Trainer:copyInputs(sample)
	-- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
	-- if using DataParallelTable. The target is always copied to a CUDA tensor
	self.input = self.input or torch.CudaTensor()
	self.target = self.target or (torch.CudaLongTensor and torch.CudaLongTensor() or torch.CudaTensor())
	self.input:resize(sample.input:size()):copy(sample.input)
	self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
	-- Training schedule
	if self.opt.decay == 'default' then
		local decay = 0
		if self.opt.dataset == 'imagenet' then
			decay = math.floor((epoch - 1) / 30)
		elseif self.opt.dataset == 'cifar10' then
			decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
		elseif self.opt.dataset == 'cifar100' then
			decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
		end
		return self.opt.lr * math.pow(0.1, decay)
	else
		local decay = math.floor((epoch - 1) / self.opt.decay_every)
		return self.opt.lr * math.pow(self.opt.decay_factor, decay)
	end
end

return M.Trainer
