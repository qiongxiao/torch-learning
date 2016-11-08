--[[
--
--  code adaption from https://github.com/facebook/fb.resnet.torch/blob/master/dataloader.lua
--
--  Single-threaded/Multi-threaded data loader
--
--]]

local datasets = require 'datasets.init'
local Scorer = require 'utils.scorer'

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
	self.batchsize = opt.batchsize

	--  single-threaded version
	require('datasets/' .. opt.dataset)
	self.__size = dataset:size()
	self.dataset = dataset
	self.preprocess = dataset:preprocess()
	self.vocabSize = self.dataset.vocabSize
end

function DataLoader:size()
	return math.ceil(self.__size / self.batchsize)
end

function DataLoader:getVocabSize()
	return self.vocabSize
end

--[[
--	input:
--		torch.Tensor of size batchsize * seqLength
--]]
function DataLoader:decode(seq, indices)
	local batchsize, seqLength = seq:size(1), seq:size(2)-1
	local out = {}
	for i = 1, batchsize do
		local txt = ''
		for j = 1, seqLength do
			local idx = seq[i][j]
			-- for decode: model output range from 1 to vocabiSize+1. If output is vocabiSize+1 whose devocab should be nil, it means it is the end sign
			local word = self.dataset.devocab[idx]
			if not word then break end
			if j >= 2 then txt = txt .. ' ' end
			txt = txt .. word
		end
		local path = self.dataset:getPath(indices[i])
		table.insert(out, {file_path = path, caption = txt})
	end
	return out
end

function DataLoader:scorerInit(scorerTypes)
	self.scorers = {}
	for _, v in scorerTypes do
		table.insert(self.scorers, Scorer(v))
	end
end

function DataLoader:scorerUpdate(seq, indices)
	local batchsize = seq:size(1)
	for i = 1, batchsize do
		local captions = self.dataset:getCaptions(indices[i])
		for _, scorer in self.scorers do
			scorer:update(captions, seq[i])
		end
	end
end

function DataLoader:scorerCompute()
	local s = {}
	for _, scorer in self.scorers do
		local sa, ss = scorer:computeScore()
		s[scorer.type] = {score=sa, scores=ss}
	end
	return s
end

function DataLoader:run()
	local size, batchsize = self.__size, self.batchsize
	-- randomize the order of data
	local perm = torch.randperm(size)
	
	local idx, sample = 1, nil

	local function makebatches()
		local seqPerImg, seqLength = self.seqPerImg, self.seqLength
		
		if idx <= size then
			-- choose indices inside batch
			local indices = perm:narrow(1, idx, math.min(batchsize, size - idx + 1)):long()
			local sz = indices:size(1)

			local batch, imageSize
			local target = torch.IntTensor(sz * seqPerImg, seqLength+1)
			-- insert the start sign (vocabSize+1)
			target[{{},{1}}]:fill(self.vocabSize+1)

			for i, idx in ipairs(indices:totable()) do
				local sample = self.dataset:get(idx)

				-- fetch image
				local input = self.preprocess(sample.input)
				if not batch then
					imageSize = input:size():totable()
					batch = torch.FloatTensor(sz, table.unpack(imageSize))
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
				input = batch,
				-- output target size is batchsize * (seqLength + 1)
				target = target,
				index = indices
			}
		else
			return nil
		end
	end

	local n = 0
	local function loop()
		sample = makebatches()
		if not sample then
			return nil
		end
		n = n + 1
		return n, sample
	end

	return loop
end

return M.DataLoader
