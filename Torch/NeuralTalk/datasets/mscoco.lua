--[[
-- some code from "https://raw.githubusercontent.com/facebook/fb.resnet.torch/master/datasets/imagenet.lua"
--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  MsCoco dataset loader
--
--]]

local image = require 'image'
local paths = require 'paths'
local ffi = require 'ffi'

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
end

function MscocoDataset:__init(imageInfo, opt, split)
	self.imageInfo = imageInfo[split]
	rebuiltCaption(self.imageInfo, imageInfo['vocab'], opt)
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

