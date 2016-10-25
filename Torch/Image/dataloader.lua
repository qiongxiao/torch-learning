--[[ code from https://github.com/facebook/fb.resnet.torch/blob/master/dataloader.lua
--
--  Single-threaded data loader
--]]

local datasets = require 'datasets/init'
local M = {}
local DataLoader = torch.class('cnn.DataLoader', M)

function DataLoader.create(opt)
	-- The train and val loader
	local loaders = {}
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

	return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
	require('datasets/' .. opt.dataset)
	self.nCrops = (split == 'val' and opt.tenCrop) and 10 or 1
	self.__size = dataset:size()
	self.batchsize = math.floor(opt.batchsize / self.nCrops)
	self.split = split
	self.opt = opt
	self.dataset = dataset
	self.preprocess = dataset:preprocess()
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchsize)
end

function DataLoader:run()
	local size, batchsize = self.__size, self.batchsize
	-- randomize the order of data
	local perm = torch.randperm(size)
	
	local idx, sample = 1, nil
	local function makebatches(idx)
		if idx <= size then
			-- choose indices inside batch
			local indices = perm:narrow(1, idx, math.min(batchsize, size - idx + 1))
			if self.opt.dataAug == 0 then
				local sample = self.dataset:get({indices:totable()})
				local input = self.preprocess(sample.input)
				collectgarbage()
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
				return {
					input = batch:view(sz * nCrops, table.unpack(imageSize)),
					target = target,
				}
			end
			idx = idx + batchsize
		else
			return nil
		end
	end

	local n = 0
	local function loop()
		sample = makebatches(idx)
		if sample then
			return nil
		end
		n = n + 1
		return n, sample
	end

	return loop
end

return M.DataLoader
