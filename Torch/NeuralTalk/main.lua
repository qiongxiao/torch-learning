--[[
--
--  some code from https://github.com/facebook/fb.resnet.torch/blob/master/main.lua
--
--]]
require 'torch'
require 'cutorch'
require 'paths'
require 'optim'
require 'nn'

local opts = require 'opts'
local DataLoader = require 'dataloader'
local models = require 'models.init'
local Trainer = require 'trainer'
local checkpoints = require 'utils.checkpoints'
local Plotter = require 'utils.plotter'
local utils = require 'utils.utils'

torch.setdefaulttensortype('torch.FloatTensor')

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.loadLatestInfo(opt)
local plotter = Plotter(opt)

-- Data loading
local trainLoader, valLoader, testLoader = DataLoader.create(opt)

local vocabSize = trainLoader:getVocabSize()

-- Create model
local model, criterion = models.setup(opt, vocabSize, checkpoint)


-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
	local loader
	if opt.dataset == 'flickr8k' then
		loader = testLoader
	else
		loader = valLoader
	end
	local testLoss, out = trainer:test(0, loader)
	print('<Testing> * Loss:', testLoss)
	utils.writeJson('predict_caption.json', out)
	return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or 1
local bestLoss = math.huge
for epoch = startEpoch, opt.maxEpochs do
	-- Train for a single epoch

	local finetune = false
	if opt.finetuneAfter > 0 and epoch >= opt.finetuneAfter then
		finetune = true
	end

	local trainLoss
	trainLoss = trainer:train(epoch, trainLoader, finetune, plotter)

	-- Run model on validation set
	local testLoss, _ = trainer:test(epoch, valLoader)

	local bestModel = false
	if testLoss < bestLoss then
		bestModel = true
		bestLoss = testLoss
		print('<Training> * Best model Loss:', testLoss)
	end
	collectgarbage()
	
	if opt.checkEvery > 0 and epoch % opt.checkEvery == 0 then
		checkpoints.saveModel(epoch, model, trainer.optimConfig, bestModel, opt)
		plotter:checkpoint()
		if opt.maxCheckpointsNum > 0 and (epoch/opt.checkEvery) > opt.maxCheckpointsNum then
			checkpoints.cleanModel(epoch - opt.checkEvery*opt.maxCheckpointsNum, opt)
		end
	end
	
	print("plot")
	plotter:add('Train Loss - Epoch', 'Train', epoch, trainLoss)
	plotter:add('Loss', 'Train', epoch, trainLoss)
	plotter:add('Loss', 'Validation', epoch, testLoss)

	collectgarbage()
end

if (opt.dataset == 'flickr8k') then
	local testLoss, out = trainer:test(0, testLoader)
	print('<Testing> * Loss:', testLoss)
	utils.writeJson('test_predict_caption.json', out)
end
