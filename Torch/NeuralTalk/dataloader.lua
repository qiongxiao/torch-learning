--[[ code from https://github.com/facebook/fb.resnet.torch/blob/master/dataloader.lua
--
--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Single-threaded/Multi-threaded data loader
--
--]]

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('cnn.DataLoader', M)

function DataLoader.create(opt)
	-- The train and val loader
	local loaders = {}

	if opt.dataset == 'MNIST' then
		for i, split in ipairs{'train', 'val', 'test'} do
			local config = {
				opt = opt,
				split = split,
			}
			local dataset = datasets.create(config)
			loaders[i] = M.DataLoader(dataset, opt, split)
		end
	else
		local yuvkernel = nil
		if opt.colorspace == 'yuv' then
			require 'torch'
			require 'nn'
			yuvkernel = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
		end
		for i, split in ipairs{'train', 'val'} do
			local config = {
				opt = opt,
				split = split,
				yuvkernel = yuvkernel
			}
			local dataset = datasets.create(config)
			loaders[i] = M.DataLoader(dataset, opt, split)
		end
	end
	return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
	self.nThreads = opt.nThreads
	self.dataAug = opt.dataAug
	self.nCrops = (split ~= 'train' and opt.tenCrop) and 10 or 1
	self.batchsize = math.floor(opt.batchsize / self.nCrops)

	--  single-threaded version
	if (opt.nThreads == 1) then
		require('datasets/' .. opt.dataset)
		self.__size = dataset:size()
		self.dataset = dataset
		self.preprocess = dataset:preprocess()
	else
		local manualSeed = opt.manualSeed
		local function init()
			require('datasets/' .. opt.dataset)
		end
		local function main(idx)
			if manualSeed ~= 0 then
				torch.manualSeed(manualSeed + idx)
			end
			torch.setnumthreads(1)
			_G.dataset = dataset
			_G.preprocess = dataset:preprocess()
			return dataset:size()
		end

		local threads, sizes = Threads(opt.nThreads, init, main)
		self.threads = threads
		self.__size = sizes[1][1]
	end
end

function DataLoader:size()
	return math.ceil(self.__size / self.batchsize)
end

function DataLoader:run()
	local size, batchsize = self.__size, self.batchsize
	-- randomize the order of data
	local perm = torch.randperm(size)
	
	local idx, sample = 1, nil

	local function makebatches()
		if idx <= size then
			-- choose indices inside batch
			local indices = perm:narrow(1, idx, math.min(batchsize, size - idx + 1)):long()
			if self.dataAug == 0 then
				local sample = self.dataset:getbatch(indices)
				local input = self.preprocess(sample.input)
				collectgarbage()
				idx = idx + batchsize
				return {
					input = input,
					target = sample.target
				}
			else
				local sz = indices:size(1)
				local batch, imageSize
				local target = torch.IntTensor(sz)
				for i, idx in ipairs(indices:totable()) do
					local sample = self.dataset:get(idx)
					local input = self.preprocess(sample.input)
					if not batch then
						imageSize = input:size():totable()
						if nCrops > 1 then table.remove(imageSize, 1) end
						batch = torch.FloatTensor(sz, nCrops, table.unpack(imageSize))
					end
					batch[i]:copy(input)
					target[i] = sample.target
				end
				collectgarbage()
				idx = idx + batchsize
				return {
					input = batch:view(sz * nCrops, table.unpack(imageSize)),
					target = target,
				}
			end
		else
			return nil
		end
	end

	local function enqueue()
		while idx <= size and threads:acceptsjob() do
			local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
			if self.dataAug == 0 then
				threads:addjob(
					function(indices)
						local sample = self.dataset:getbatch(indices)
						local input = self.preprocess(sample.input)
						collectgarbage()
						return {
							input = input,
							target = sample.target
						}
					end,
					function(_sample_)
						sample = _sample_
					end,
					indices
				)
			else
				threads:addjob(
					function(indices, nCrops)
						local sz = indices:size(1)
						local batch, imageSize
						local target = torch.IntTensor(sz)
						for i, idx in ipairs(indices:totable()) do
							local sample = _G.dataset:get(idx)
							local input = _G.preprocess(sample.input)
							if not batch then
								imageSize = input:size():totable()
								if nCrops > 1 then table.remove(imageSize, 1) end
								batch = torch.FloatTensor(sz, nCrops, table.unpack(imageSize))
							end
							batch[i]:copy(input)
							target[i] = sample.target
						end
						collectgarbage()
						return {
							input = batch:view(sz * nCrops, table.unpack(imageSize)),
							target = target,
						}
					end,
					function(_sample_)
						sample = _sample_
					end,
					indices,
					self.nCrops
				)
			end
			idx = idx + batchSize
		end
	end


	local n = 0
	local function loop()
		if self.nThreads == 1 then
			sample = makebatches()
			if not sample then
				return nil
			end
			n = n + 1
		else
			enqueue()
			if not threads:hasjob() then
				return nil
			end
			threads:dojob()
			if threads:haserror() then
				threads:synchronize()
			end
			enqueue()
			n = n + 1			
		end
		return n, sample
	end

	return loop
end

return M.DataLoader
