--[[
--
--  code from https://github.com/facebook/fb.resnet.torch/blob/master/models/init.lua
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--
--]]
require 'nn'
require 'cunn'
require 'cudnn'

local M = {}

function M.setup(opt, checkpoint)
	local model
	if checkpoint then
		local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
		assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
		print('<model init> => Resuming model from ' .. modelPath)
		model = torch.load(modelPath):cuda()
	elseif opt.retrain ~= 'none' then
		assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
		print('<model init> => Loading model from file: ' .. opt.retrain)
		model = torch.load(opt.retrain):cuda()
	else
		print('<model init> => Creating model from file: models/' .. opt.netType .. '.lua')
		model = require('models/' .. opt.netType)(opt)
	end

	-- First remove any DataParallelTable
	if torch.type(model) == 'nn.DataParallelTable' then
		model = model:get(1)
	end

	-- optnet is an general library for reducing memory usage in neural networks
	if opt.optnet then
		local optnet = require 'optnet'
		local imsize = opt.dataset == 'imagenet' and 224 or 32
		local sampleInput = torch.zeros(4,3,imsize,imsize):cuda()
		optnet.optimizeMemory(model, sampleInput, {inplace = false, mode = 'training'})
	end

	-- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
	-- containers override backwards to call backwards recursively on submodules
	if opt.shareGradInput then
		M.shareGradInput(model)
	end

	-- For resetting the classifier when fine-tuning on a different Dataset
	if opt.resetClassifier and not checkpoint then
		print('<model init>  => Replacing classifier with ' .. opt.nClasses .. '-way classifier')

		local orig = model:get(#model.modules)
		assert(torch.type(orig) == 'nn.Linear', 'expected last layer to be fully connected')

		local linear = nn.Linear(orig.weight:size(2), opt.nClasses)
		linear.bias:zero()

		model:remove(#model.modules)
		model:add(linear:cuda())
   end

	-- Set the CUDNN flags
	cudnn.fastest = true
	cudnn.benchmark = true

   -- Wrap the model with DataParallelTable, if using more than one GPU
	if opt.nGPU > 1 then
		local gpus = torch.range(1, opt.nGPU):totable()
		local fastest, benchmark = cudnn.fastest, cudnn.benchmark

		local dpt = nn.DataParallelTable(1, true, true)
			:add(model, gpus)
			:threads(function()
						local cudnn = require 'cudnn'
						cudnn.fastest, cudnn.benchmark = fastest, benchmark
					end)
		dpt.gradInput = nil

		model = dpt:cuda()
	end

	local criterion = nn.CrossEntropyCriterion():cuda()
	return model, criterion
end

function M.shareGradInput(model)
	local function sharingKey(m)
		local key = torch.type(m)
		if m.__shareGradInputKey then
			key = key .. ':' .. m.__shareGradInputKey
		end
		return key
	end

	-- Share gradInput for memory efficient backprop
	local cache = {}
	model:apply(function(m)
			local moduleType = torch.type(m)
			if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
				local key = sharingKey(m)
				if cache[key] == nil then
					cache[key] = torch.CudaStorage(1)
				end
				m.gradInput = torch.CudaTensor(cache[key], 1, 0)
			end
		end)
	for i, m in ipairs(model:findModules('nn.ConcatTable')) do
		if cache[i % 2] == nil then
			cache[i % 2] = torch.CudaStorage(1)
		end
		m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
	end
end

return M
