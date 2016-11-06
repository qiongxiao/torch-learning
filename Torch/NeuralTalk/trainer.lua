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

function Trainer:__init(model, criterion, opt, optimConfig)
	self.cnn, self.feature2seq = model.cnn, model.feature2seq
	self.criterion = criterion

	self.lstmOptimizer = opt.optimizer
	local lstmOptimConfig
	if self.lstmOptimizer == 'sgd' then
		lstmOptimConfig = optimConfig and optimConfig.lstm or {
			learningRate = opt.lr,
			learningRateDecay = opt.lr_decay,
			weigthDecay = opt.weigthDecay,
			momentum = opt.momentum,
			nesterov = true,
			dampening = 0.0,

		}
	elseif self.lstmOptimizer == 'adam' then
		lstmOptimConfig = optimConfig and optimConfig.lstm or {
			learningRate = opt.lr,
			learningRateDecay = opt.lr_decay,
			weigthDecay = opt.weigthDecay,
			beta1 = opt.optimAlpha,
			beta2 = opt.optimBeta,
		}
	else
		error('invalid optimizer ' .. opt.optimizer)
	end

	self.cnnOptimizer = opt.cnnOptimizer
	local cnnOptimConfig
	if self.cnnOptimizer == 'sgd' then
		cnnOptimConfig = optimConfig and optimConfig.cnn or {
			learningRate = opt.cnnLr,
			learningRateDecay = opt.cnnLr_decay,
			weigthDecay = opt.cnnWeigthDecay,
			momentum = opt.cnnMomentum,
			nesterov = true,
			dampening = 0.0,
		}
	elseif self.cnnOptimizer == 'adam' then
		cnnOptimConfig = optimConfig and optimConfig.cnn or {
			learningRate = opt.cnnLr,
			learningRateDecay = opt.cnnLr_decay,
			weigthDecay = opt.cnnWeigthDecay,
			beta1 = opt.cnnOptimAlpha,
			beta2 = opt.cnnOptimBeta,
		}
	else
		error('invalid cnnOptimizer ' .. opt.cnnOptimizer)
	end

	self.optimConfig = {lstm = lstmOptimConfig, cnn = cnnOptimConfig}
	self.opt = opt
	self.params, self.gradParams = self.feature2seq:getParameters()
	self.cnnParams, self.cnnGradParams = self.cnn:getParameters()
end

function Trainer:train(epoch, dataloader, finetune, plotter)
	-- Trains the model for a single epoch
	self.optimConfig.lstm.learningRate = self:learningRate(epoch)
	self.optimConfig.cnn.learningRate = self:learningRate(epoch, 'cnn')

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
	self.cnn:training()
	self.feature2seq:training()

	for n, sample in dataloader:run() do
		local dataTime = dataTimer:time().real

		local batchsize = sample.input:size(1)
		-- Copy input and target to the GPU
		self:copyInputs(sample)

		self.cnn:forward(self.input)

		self.feature2seq:forward{self.cnn.output, self.target}
		local loss = self.criterion:forward(self.feature2seq.output, self.target)

		self.feature2seq:zeroGradParameters()
		if finetune then
			self.cnn:zeroGradParameters()
		end

		self.criterion:backward(self.feature2seq.output, self.target)
		self.feature2seq:backward({self.cnn.output, self.target}, self.criterion.gradInput)

		self.params:clamp(-self.opt.gradClip, self.opt.gradClip)

		if finetune then
			self.cnn:backward(self.input, self.feature2seq.gradInput[1])
			self.cnnParams:clamp(-self.opt.gradClip, self.opt.gradClip)
		end

		if self.lstmOptimizer == 'sgd' then
			optim.sgd(feval, self.params, self.optimConfig.lstm)
		else
			optim.adam(feval, self.params, self.optimConfig.lstm)
		end

		if finetune then
			if self.cnnOptimizer == 'sgd' then
				optim.sgd(cnnFeval, self.cnnParams, self.optimConfig.cnn)
			else
				optim.adam(cnnFeval, self.cnnParams, self.optimConfig.cnn)
			end
		end

		lossSum = lossSum + loss*batchsize
		N = N + batchsize

		print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Loss %1.4f'):format(
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

	collectgarbage()
	return lossSum / N
end

function Trainer:test(epoch, dataloader)

	local timer = torch.Timer()
	local dataTimer = torch.Timer()
	local size = dataloader:size()

	local lossSum = 0.0
	local N = 0
	local out = {}
	
	self.cnn:evaluate()
	self.feature2seq:evaluate()

	for n, sample in dataloader:run() do
		local dataTime = dataTimer:time().real

		local batchsize = sample.input:size(1)
		-- Copy input and target to the GPU
		self:copyInputs(sample)

		self.cnn:forward(self.input)
		self.feature2seq:forward{self.cnn.output, self.target}
		local loss = self.criterion:forward(self.feature2seq.output, self.target)

		lossSum = lossSum + loss*batchsize
		N = N + batchsize

		local seq = self.feature2seq:inference(self.cnn.output)
		seq = dataloader:decode(seq, sample.path)
		
		if self.opt.verbose then
			print(seq[1].caption)
		end
		
			table.insert(out, v)
		end
		
		print((' | eval: [%d][%d/%d]    Time %.3f  Data %.3f  Loss %1.4f'):format(
			epoch, n, size, timer:time().real, dataTime, loss))

		timer:reset()
		dataTimer:reset()
	end

	self.cnn:training()
	self.feature2seq:training()

	print((' * Finished epoch # %d    Err: %7.3f\n'):format(epoch, lossSum / N))

	collectgarbage()
	return lossSum / N, out
end

function Trainer:copyInputs(sample)
	if self.opt.nGPU < 1 then
		self.input = sample.input
		self.target = sample.target
	else
		-- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
		-- if using DataParallelTable. The target is always copied to a CUDA tensor
		self.input = self.input or (self.opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
		self.target = self.target or (torch.CudaLongTensor and torch.CudaLongTensor()or torch.CudaTensor())
		self.input:resize(sample.input:size()):copy(sample.input)
		self.target:resize(sample.target:size()):copy(sample.target)
	end
end

function Trainer:learningRate(epoch, mode)
	if self.opt.decay == 'default' then
		local decay = 0
		if self.opt.dataset == 'mscoco' then
			decay = math.floor((epoch - 1) / 30)
		elseif self.opt.dataset == 'flickr8k' then
			decay = math.floor((epoch - 1) / 100)
		end
		if mode == 'cnn' then
			return self.opt.cnnLr * math.pow(0.5, decay)
		else
			return self.opt.lr * math.pow(0.5, decay)
		end
	else
		local decay = math.floor((epoch - 1) / self.opt.decay_every)
		if mode == 'cnn' then 
			return self.opt.cnnLr * math.pow(self.opt.decay_factor, decay)
		else
			return self.opt.lr * math.pow(self.opt.decay_factor, decay)
		end
	end

end

return M.Trainer
