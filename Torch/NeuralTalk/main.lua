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
if opt.nGPU > 0 then
	cutorch.manualSeedAll(opt.manualSeed)
	cutorch.setDevice(opt.nGPU)
end

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.loadCheckpointInfo(opt)
local plotter = Plotter(opt, checkpoint)

-- Data loading
local trainLoader, valLoader, testLoader = DataLoader.create(opt)

local vocabSize = trainLoader:getVocabSize()

-- Create model
local model, criterion = models.setup(opt, vocabSize, checkpoint)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

-- where to save the result
local resultDir = paths.concat(opt.save, 'result')
if not paths.dirp(resultDir) then
	os.execute('mkdir ' .. resultDir)
end

if opt.testOnly then
	local loader
	if opt.dataset == 'flickr8k' then
		loader = testLoader
	else
		loader = valLoader
	end
	local testLoss, out, scores = trainer:test(0, loader)
	utils.writeJson(paths.concat(resultDir, 'predict_caption.json'), out)
	utils.writeJson(paths.concat(resultDir, 'predict_scores.json'), scores)
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
	utils.writeJson(paths.concat(resultDir, 'val_predict_caption_' .. epoch .. '.json'), out)
	utils.writeJson(paths.concat(resultDir, 'val_predict_scores_' .. epoch .. '.json'), scores)
	
	local bestModel = false
	if scores['CIDEr']['score'] > bestLoss then
		bestModel = true
		bestLoss = scores['CIDEr']['score']
		print('<Training> * Best model CIDEr:', scores['CIDEr']['score'])
		utils.writeJson(paths.concat(resultDir, 'val_best_predict_caption.json'), out)
		utils.writeJson(paths.concat(resultDir, 'val_best_predict_scores.json'), scores)
	end
	collectgarbage()
	
	print("plot")
	plotter:add('Train Loss - Epoch', 'Train', epoch, trainLoss)
	plotter:add('Loss', 'Train', epoch, trainLoss)
	plotter:add('Loss', 'Validation', epoch, testLoss)
	plotter:add('Score - CIDEr', 'CIDEr', epoch, scores['CIDEr']['score'])
	plotter:add('Score - BLEU1', 'BLEU_1', epoch, scores['BLEU']['score'][1])
	plotter:add('Score - BLEU2', 'BLEU_2', epoch, scores['BLEU']['score'][2])
	plotter:add('Score - BLEU3', 'BLEU_3', epoch, scores['BLEU']['score'][3])
	plotter:add('Score - BLEU4', 'BLEU_4', epoch, scores['BLEU']['score'][4])
	
	if opt.checkEvery > 0 and epoch % opt.checkEvery == 0 then
		if opt.maxCheckpointsNum > 0 and (epoch/opt.checkEvery) > opt.maxCheckpointsNum then
			checkpoints.cleanModel(epoch - opt.checkEvery*opt.maxCheckpointsNum, opt)
		end
		checkpoints.saveModel(epoch, model, trainer.optimConfig, bestModel, bestLoss, opt)
		plotter:checkpoint()
	end

	collectgarbage()
end

if (opt.dataset == 'flickr8k') then
	local testLoss, out, scores = trainer:test(0, testLoader)
	utils.writeJson(paths.concat(resultDir, 'test_predict_caption.json'), out)
	utils.writeJson(paths.concat(resultDir, 'test_predict_scores.json'), scores)
end
