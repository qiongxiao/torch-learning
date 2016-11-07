--[[
--
--  some code from "https://raw.githubusercontent.com/facebook/fb.resnet.torch/master/datasets/imagenet.lua"
--
--  mscoco dataset loader
--
--]]

local image = require 'image'
local paths = require 'paths'
local ffi = require 'ffi'

local t = require 'utils.imgTransform'

local M = {}
local MscocoDataset = torch.class('NeuralTalk.MscocoDataset', M)

local function rebuiltCaption(data, vocab, opt)
	local seqLength = opt.seqLength
	local countThr = opt.wordCountThreshold
	local goodWordCount = 0
	for _, v in pairs(vocab) do
		if v['count'] > countThr then
			goodWordCount = goodWordCount + 1
		end
	end
	data.imageCaptions = data.imageCaptions:clamp(1, goodWordCount+1):narrow(2, 1, seqLength)
	return goodWordCount+1
end

function MscocoDataset:__init(imageInfo, opt, split)
	self.imageInfo = imageInfo[split]
	self.vocab = imageInfo.vocab
	self.devocab = imageInfo.devocab
	self.vocabSize = rebuiltCaption(self.imageInfo, imageInfo['vocab'], opt)
	self.opt = opt
	self.split = split
	self.dir = paths.concat(opt.data, split .. "2014")
	assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function MscocoDataset:get(i)
	local path = ffi.string(self.imageInfo.imagePath[i]:data())

	local image = self:_loadImage(paths.concat(self.dir, path))
	local startIdx = self.imageInfo.imageCapIdx[i][1]
	local endIdx = self.imageInfo.imageCapIdx[i][2]
	local captions = self.imageInfo.imageCaptions:sub(startIdx, endIdx)

	return {
		input = image,
		target = captions,
	}
end

function MscocoDataset:_loadImage(path)
	local ok, input = pcall(function()
		return image.load(path, 3, 'float')
	end)

	-- Sometimes image.load fails because the file extension does not match the
	-- image format. In that case, use image.decompress on a ByteTensor.
	if not ok then
		local f = io.open(path, 'r')
		assert(f, 'Error reading: ' .. tostring(path))
		local data = f:read('*a')
		f:close()

		local b = torch.ByteTensor(string.len(data))
		ffi.copy(b:data(), data, b:size(1))

		input = image.decompress(b, 3, 'float')
	end

	return input
end

function MscocoDataset:size()
	return self.imageInfo.imageCapIdx:size(1)
end

local meanstd = {
	mean = { 0.485, 0.456, 0.406 },
	std = { 0.229, 0.224, 0.225 },
}
local pca = {
	eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
	eigvec = torch.Tensor{
		{ -0.5675,  0.7192,  0.4009 },
		{ -0.5808, -0.0045, -0.8140 },
		{ -0.5836, -0.6948,  0.4203 },
	},
}
function ImagenetDataset:preprocess()
	if self.opt.cnnType == 'resnet' then
		if self.split == 'train' then
			return t.Compose{
				t.RandomSizedCrop(224),
				t.ColorJitter({
					brightness = 0.4,
					contrast = 0.4,
					saturation = 0.4,
				}),
				t.Lighting(0.1, pca.eigval, pca.eigvec),
				t.ColorNormalize(meanstd, self.opt.cnnType),
				t.HorizontalFlip(0.5),
			}
		elseif self.split == 'val' then
			return t.Compose{
				t.Scale(256),
				t.ColorNormalize(meanstd, self.opt.cnnType),
				t.CenterCrop(224),
			}
		else
			error('invalid split: ' .. self.split)
		end
	elseif self.opt.cnnType == 'vgg' then
		if self.split == 'train' then
			return t.Compose{
				t.RandomSizedCrop(224),
				t.ColorNormalize(meanstd, self.opt.cnnType),
				t.HorizontalFlip(0.5),
			}
		elseif self.split == 'val' then
			return t.Compose{
				t.Scale(256),
				t.ColorNormalize(meanstd, self.opt.cnnType),
				t.CenterCrop(224),
			}
		else
			error('invalid split: ' .. self.split)
		end		
	else
		error('invalid cnn_type: ' .. self.opt.cnnType)
	end
end

return M.MscocoDataset