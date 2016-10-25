--[[ code from https://github.com/facebook/fb.resnet.torch/blob/master/main.lua
--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--]]
require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'cutorch'
require 'cunn'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'
local Plotter = require 'plotter'

torch.setdefaulttensortype('torch.FloatTensor')

local opt = opts.parse(arg)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.loadLatestInfo(opt)
local plotter = Plotter(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader, testLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
	local top1Err, top5Err = trainer:test(0, valLoader)
	print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
	return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.startEpoch
local bestTop1 = math.huge
local bestTop5 = math.huge
for epoch = startEpoch, opt.maxEpochs do
	-- Train for a single epoch
	local trainTop1, trainTop5, trainLoss
	if opt.plotEvery == 1 then
		trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader, plotter)
	else
		trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)
	end

	-- Run model on validation set
	local testTop1, testTop5, testLoss = trainer:test(epoch, valLoader)

	local bestModel = false
	if testTop1 < bestTop1 then
		bestModel = true
		bestTop1 = testTop1
		bestTop5 = testTop5
		print('<Training> * Best model ', testTop1, testTop5)
	end
	
	if epoch % opt.checkEvery == 0 then
		checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
		plotter:checkpoint()
	end
	
	plotter:add('Train Loss / Epoch', 'Train', epoch, trainLoss)
	plotter:add('Loss', 'Train', epoch, trainLoss)
	plotter:add('Accuracy', 'Validation', epoch, testTop1)
	plotter:add('Loss', 'Validation', epoch, testLoss)
end

print(string.format('<Training> * Finished top1: %6.3f  top5: %6.3f', bestTop1, bestTop5))

if opt.dataset == 'MNIST' then
	local testTop1, testTop5, _ = trainer:test(epoch, testLoader)
	print(string.format('<Testing> * Finished top1: %6.3f  top5: %6.3f', testTop1, testTop5))
end