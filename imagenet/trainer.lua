--[[
--
--  code adaption from https://github.com/facebook/fb.resnet.torch/blob/master/train.lua
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

function Trainer:__init(model, criterion, opt, optimConfig)
	self.model = model
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
	self.opt = opt
	self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader, plotter)
	-- Trains the model for a single epoch
	self.optimConfig.learningRate = self:learningRate(epoch)

	local timer = torch.Timer()
	local dataTimer = torch.Timer()

	local function feval()
		return self.criterion.output, self.gradParams
	end

	local trainSize = dataloader:size()
	local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
	local N = 0

	print('<Training> => Training epoch # ' .. epoch)
	-- set the batch norm to training mode
	self.model:training()
	for n, sample in dataloader:run() do
		local dataTime = dataTimer:time().real

		-- Copy input and target to the GPU
		self:copyInputs(sample)

		local output = self.model:forward(self.input):float()
		local batchsize = output:size(1)
		local loss = self.criterion:forward(self.model.output, self.target)

		self.model:zeroGradParameters()
		self.criterion:backward(self.model.output, self.target)
		self.model:backward(self.input, self.criterion.gradInput)
		
		if self.optimizer == 'sgd' then
			optim.sgd(feval, self.params, self.optimConfig)
		else
			optim.adam(feval, self.params, self.optimConfig)
		end

		local top1, top5 = self:computeScore(output, sample.target, 1)
		top1Sum = top1Sum + top1*batchsize
		top5Sum = top5Sum + top5*batchsize
		lossSum = lossSum + loss*batchsize
		N = N + batchsize

		print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top5 %7.3f'):format(
			epoch, n, trainSize, timer:time().real, dataTime, loss, top1, top5))

		-- check that the storage didn't get changed do to an unfortunate getParameters call
		assert(self.params:storage() == self.model:parameters()[1]:storage())

		timer:reset()
		dataTimer:reset()
		if self.opt.plotEvery == 1 then
			plotter:add('Train Loss - Iteration', 'Train', (epoch-1)*trainSize+n, loss)
		end
	end
	return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:test(epoch, dataloader)
	-- Computes the top-1 and top-5 err on the validation set

	local timer = torch.Timer()
	local dataTimer = torch.Timer()
	local size = dataloader:size()

	local nCrops = self.opt.tenCrop and 10 or 1
	local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
	local N = 0

	self.model:evaluate()
	for n, sample in dataloader:run() do
		local dataTime = dataTimer:time().real

		-- Copy input and target to the GPU
		self:copyInputs(sample)

		local output = self.model:forward(self.input)
		local nOutput = output
		if nCrops > 1 then
			-- Sum over crops
			nOutput = output:view(output:size(1) / nCrops, nCrops, output:size(2))
			:sum(2):squeeze(2)
		end

		local batchsize = nOutput:size(1)
		local loss = self.criterion:forward(nOutput, self.target)

		local top1, top5 = self:computeScore(nOutput:float(), sample.target, nCrops)
		top1Sum = top1Sum + top1*batchsize
		top5Sum = top5Sum + top5*batchsize
		lossSum = lossSum + loss*batchsize
		N = N + batchsize

		print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
			epoch, n, size, timer:time().real, dataTime, top1, top1Sum / N, top5, top5Sum / N))

		timer:reset()
		dataTimer:reset()
	end
	self.model:training()

	print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n'):format(
		epoch, top1Sum / N, top5Sum / N))

	return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:computeScore(output, target, nCrops)
	-- Computes the top1 and top5 error rate
	local batchsize = output:size(1)

	local _ , predictions = output:float():sort(2, true) -- descending (predictions is matrix of indices)

	-- Find which predictions match the target
	local correct = predictions:eq(
		target:long():view(batchsize, 1):expandAs(output))

	-- Top-1 score
	local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchsize)

	-- Top-5 score, if there are at least 5 classes
	local len = math.min(5, correct:size(2))
	local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchsize)

	return top1 * 100, top5 * 100
end

function Trainer:copyInputs(sample)
	-- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
	-- if using DataParallelTable. The target is always copied to a CUDA tensor
	self.input = self.input or (self.opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
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
