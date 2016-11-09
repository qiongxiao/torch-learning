--[[
--
--  code adaption from https://github.com/facebook/fb.resnet.torch/blob/master/main.lua
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

if not paths.dirp('result') then
	os.execute('mkdir result')
end

if opt.testOnly then
	local loader
	if opt.dataset == 'flickr8k' then
		loader = testLoader
	else
		loader = valLoader
	end
	local testLoss, out, scores = trainer:test(0, loader)
	utils.writeJson('result/predict_caption.json', out)
	utils.writeJson('result/predict_scores.json', scores)
	return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.startEpoch
local bestLoss = checkpoint and checkpoint.bestLoss or 0
for epoch = startEpoch, opt.maxEpochs do
	local finetune = false
	if opt.finetuneAfter > 0 and epoch >= opt.finetuneAfter then
		finetune = true
	end

	-- Train for a single epoch
	local trainLoss
	trainLoss = trainer:train(epoch, trainLoader, finetune, plotter)

	-- Run model on validation set
	local testLoss, out, scores = trainer:test(epoch, valLoader)
	utils.writeJson('result/val_predict_caption_' .. epoch .. '.json', out)
	utils.writeJson('result/val_predict_scores_' .. epoch .. '.json', scores)
	
	local bestModel = false
	if scores['CIDEr']['score'] > bestLoss then
		bestModel = true
		bestLoss = scores['CIDEr']['score']
		print('<Training> * Best model CIDEr:', scores)
	end
	collectgarbage()
	
	print("plot")
	plotter:add('Train Loss - Epoch', 'Train', epoch, trainLoss)
	plotter:add('Loss', 'Train', epoch, trainLoss)
	plotter:add('Loss', 'Validation', epoch, testLoss)
	plotter:add('CIDEr', 'Validation', epoch, scores['CIDEr']['score'])
	if opt.checkEvery > 0 and epoch % opt.checkEvery == 0 then
		checkpoints.saveModel(epoch, model, trainer.optimConfig, bestModel, bestLoss, opt)
		plotter:checkpoint()
		if opt.maxCheckpointsNum > 0 and (epoch/opt.checkEvery) > opt.maxCheckpointsNum then
			checkpoints.cleanModel(epoch - opt.checkEvery*opt.maxCheckpointsNum, opt)
		end
	end

	collectgarbage()
end

if (opt.dataset == 'flickr8k') then
	local testLoss, out, scores = trainer:test(0, testLoader)
	utils.writeJson('result/test_predict_caption.json', out)
	utils.writeJson('result/test_predict_scores.json', scores)
end
