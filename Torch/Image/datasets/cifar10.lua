--[[
--
--  code from https://github.com/facebook/fb.resnet.torch/blob/master/datasets/cifar10.lua
--
--  CIFAR-10 dataset loader
--
--]]
local t = require 'datasets/image_transform'

local M = {}

local CifarDataset = torch.class('CifarDataset', M)

function CifarDataset:__init( imageInfo, opt, config )
	local split = config.split or nil
	self.split = split
	assert(imageInfo[split], split)
	self.imageInfo = imageInfo[split]
	self.split = split
	self.config = config
end

function CifarDataset:get(i)
	local image = self.imageInfo.data[i]:float()
	local label = self.imageInfo.labels[i]

	return {
		input = image,
		target = label,
	}
end

function CifarDataset:getbatch( i )
	local image = self.imageInfo.data:index(1, i):float()
	local label = self.imageInfo.labels:index(1, i)

	return {
		input = image,
		target = label,
	}
end

function CifarDataset:size()
	return self.imageInfo.data:size(1)
end

-- Computed from entire CIFAR-10 training set
local meanstd = {
	rgb = {
		mean = {125.3, 123.0, 113.9},
		std  = {63.0,  62.1,  66.7},
	},
	yuv = {
		mean = {0, -4.3, 2.4},
		std = {0, 13.7, 17.1},
	}
}

function CifarDataset:preprocess()
	if self.split == 'train' then
		if self.config.opt.dataAug == 0 then
			return t.ColorNormalize(meanstd, self.config)
		else
			return t.Compose{
				t.ColorNormalize(meanstd, self.config),
				t.HorizontalFlip(0.5),
				t.RandomCrop(32, 4),
			}
		end
	elseif self.split == 'val' then
		return t.ColorNormalize(meanstd, self.config)
	else
		error('invalid split: ' .. self.split)
	end
end

return M.CifarDataset
