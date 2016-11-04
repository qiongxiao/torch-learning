--[[
--
--  code from https://github.com/facebook/fb.resnet.torch/blob/master/dataloader.lua
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

	local splits

	if opt.dataset == 'flickr8k' then
		splits = {'train', 'val', 'test'}
	else
		splits = {'train', 'val'}
	end

	for i, split in pairs(splits) do
		local config = {
			opt = opt,
			split = split,
		}
		local dataset = datasets.create(config)
		loaders[i] = M.DataLoader(dataset, opt, split)
	end

	return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
	self.nThreads = opt.nThreads
	self.seqPerImg = opt.seqPerImg
	self.seqLength = opt.seqLength
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

function DataLoader:getVocabSize()
	return self.dataset.vocabSize
end

--[[
--	input:
--		torch.Tensor of size batchsize * seqLength
--]]
function DataLoader:decode(seq)
	local batchsize, seqLength = seq:size(1), seq:size(2)-1
	local out = {}
	for i = 1, batchsize do
		local txt = ''
		for j = 1, seqLength do
			local idx = seq[i][j]
			local word = self.dataset.devocab[idx]
			if not word then break end
			if j >= 2 then txt = txt .. ' ' end
			txt = txt .. word
		end
		table.insert(out, txt)
	end
	return out
end

function DataLoader:run()
	local size, batchsize = self.__size, self.batchsize
	-- randomize the order of data
	local perm = torch.randperm(size)
	
	local idx, sample = 1, nil

	local function makebatches()
		local nCrops = self.nCrops
		local seqPerImg, seqLength = self.seqPerImg, self.seqLength
		
		if idx <= size then
			-- choose indices inside batch
			local indices = perm:narrow(1, idx, math.min(batchsize, size - idx + 1)):long()
			local sz = indices:size(1)
			local batch, imageSize
			local target = torch.IntTensor(sz * seqPerImg, seqLength+1)
			target[{{},{1}}]:fill(self.dataset.vocabSize+1)
			for i, idx in ipairs(indices:totable()) do
				local sample = self.dataset:get(idx)

				-- fetch image
				local input = self.preprocess(sample.input)
				if not batch then
					imageSize = input:size():totable()
					if nCrops > 1 then table.remove(imageSize, 1) end
					batch = torch.FloatTensor(sz, nCrops, table.unpack(imageSize))
				end
				batch[i]:copy(input)

				-- fetch sequences
				local seq
				local nSeq = sample.target:size(1)
				if nSeq < seqPerImg then
					-- we need to subsample (with replacement)
					seq = torch.LongTensor(seqPerImg, seqLength)
					for q = 1, seqPerImg do
						local seqIdx = torch.random(1, nSeq)
						seq[{{q, q}}] = sample.target[{{seqIndex, seqIdx}}]
					end
				elseif nSeq > seqPerImg then
					local seqIndex = torch.random(1, nSeq - seqPerImg + 1)
					seq = sample.target:narrow(1, seqIndex, seqPerImg)
				else
					seq = sample.target
				end

				local startIdx = (i - 1) * seqPerImg
				target[{{startIdx+1, startIdx+seqPerImg},{2, 1+seqLength}}] = seq
			end
			collectgarbage()
			idx = idx + batchsize
			return {
				input = batch:view(sz * nCrops, table.unpack(imageSize)),
				target = target,
			}
		else
			return nil
		end
	end

	local function enqueue()
		while idx <= size and threads:acceptsjob() do
			local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
			threads:addjob(
				function(indices, nCrops, seqPerImg, seqLength)
					local sz = indices:size(1)
					local batch, imageSize
					local target = torch.IntTensor(sz * seqPerImg, seqLength+1)
					target[{{},{1}}]:fill(self.dataset.vocabSize+1)
					for i, idx in ipairs(indices:totable()) do
						local sample = _G.dataset:get(idx)

						-- fetch image
						local input = _G.preprocess(sample.input)
						if not batch then
							imageSize = input:size():totable()
							if nCrops > 1 then table.remove(imageSize, 1) end
							batch = torch.FloatTensor(sz, nCrops, table.unpack(imageSize))
						end
						batch[i]:copy(input)

						-- fetch sequences
						local seq
						local nSeq = sample.target:size(1)
						if nSeq < seqPerImg then
							-- we need to subsample (with replacement)
							seq = torch.LongTensor(seqPerImg, seqLength)
							for q = 1, seqPerImg do
								local seqIdx = torch.random(1, nSeq)
								seq[{{q, q}}] = sample.target[{{seqIndex, seqIdx}}]
							end
						elseif nSeq > seqPerImg then
							local seqIndex = torch.random(1, nSeq - seqPerImg + 1)
							seq = sample.target:narrow(1, seqIndex, seqPerImg)
						else
							seq = sample.target
						end

						local startIdx = (i - 1) * seqPerImg
						target[{{startIdx+1, startIdx+seqPerImg},{2, 1+seqLength}}] = seq
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
				self.nCrops,
				self.seqPerImg,
				self.seqLength
			)
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
