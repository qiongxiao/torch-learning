-- code from https://github.com/facebook/fb.resnet.torch/blob/master/datasets/cifar10.lua
local t = require 'datasets/image_transform'

local M = {}

local CifarDataset = torch.class('CifarDataset', M)

function CifarDataset:__init( imageInfo, opt, split )
	assert(imageInfo[split], split)
	self.imageInfo = imageInfo[split]
	self.split = split
	self.opt = opt
	self.colorspace = opt.colorspace
end

function CifarDataset:get(i)
	local image = self.imageInfo.data[i]:float()
	local label = self.imageInfo.labels[i]

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
	}
	yuv = {
		mean = {},
		std = {},
	}
}

function CifarDataset:preprocess_simple()
	if self.split == 'train' or self.split == 'test' or self.split == 'val' then
		return t.ColorNormalize(meanstd, self.colorspace)
	else
		error('invalid split: ' .. self.split)
	end
end

function CifarDataset:preprocess_augment()
	if self.split == 'train' then
	return t.Compose{
		t.ColorNormalize(meanstd, self.colorspace),
		t.HorizontalFlip(0.5),
		t.RandomCrop(32, 4),
	}
	elseif self.split == 'test' or self.split == 'val' then
		return t.ColorNormalize(meanstd, self.colorspace)
	else
		error('invalid split: ' .. self.split)
	end
end

return M.CifarDataset