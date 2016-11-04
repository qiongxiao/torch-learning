--[[
--
--  some code from https://github.com/facebook/fb.resnet.torch/blob/master/main.lua
--
--]]
require 'torch'
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
local checkpoint, optimState, cnnOptimState = checkpoints.loadLatestInfo(opt)
local plotter = Plotter(opt)

-- Data loading
local trainLoader, valLoader, testLoader = DataLoader.create(opt)

local vocabSize = trainLoader:getVocabSize()

-- Create model
local cnn, feature2seq, criterion = models.setup(opt, vocabSize, checkpoint)


-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(cnn, feature2seq, criterion, opt, optimState, cnnOptimState)

if opt.testOnly then
	local loader
	if opt.dataset == 'flickr8k' then
		loader = testLoader
	else
		loader = valLoader
	end
	local _, out = trainer:test(0, loader)
	--utils.writeJson('predict_caption.json', out)
	return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or 1
local bestLoss = math.huge
for epoch = startEpoch, opt.maxEpochs do
	-- Train for a single epoch
	local finetune = 0
	if opt.finetune == 1 and epoch >= opt.finetuneAfter then
		finetune = 1
	end

	local trainLoss
	trainLoss = trainer:train(epoch, trainLoader, finetune, plotter)

	-- Run model on validation set
	local testLoss, out = trainer:test(epoch, valLoader)

	local bestModel = false
	if testLoss < bestLoss then
		bestModel = true
		bestLoss = testLoss
		print('<Training> * Best model Loss:', testLoss)
	end
	
	if opt.checkEvery > 0 and epoch % opt.checkEvery == 0 then
		checkpoints.saveModel(epoch, cnn, feature2seq, trainer.optimConfig, trainer.cnnOptimConfig, bestModel, opt)
		plotter:checkpoint()
	end
	
	plotter:add('Train Loss - Epoch', 'Train', epoch, trainLoss)
	plotter:add('Loss', 'Train', epoch, trainLoss)
	plotter:add('Loss', 'Validation', epoch, testLoss)
end

if (opt.dataset == 'flickr8k') then
	local testLoss, _ = trainer:test(0, testLoader)
	print('<Testing> * Loss:', testLoss)
end
