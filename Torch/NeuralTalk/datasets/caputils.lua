local sys = require 'sys'
local ffi = require 'ffi'
local paths = require 'paths'
local utils = require 'utils.utils'
require 'torch'

local caputils = {}

function caputils.preproCaptions(data)
	local maxLength = -1
	local imageCapIdx = torch.LongTensor(#data, 2):zero()
	for i, img in pairs(data) do
		img['processed_tokens'] = {}
		local nCaps = #(img['captions'])
		for _, s in pairs(img['captions']) do
			local tokens = {}
			local txt = string.gsub(s, "%p", "")
			for token in string.gmatch(txt, "[^%s]+") do
				table.insert(tokens, token)
			end
			maxLength = math.max(maxLength, #tokens)
			table.insert(img['processed_tokens'], tokens)
		end
		img['captions'] = nil
		if i == 1 then
			imageCapIdx[1][1] = 1
			imageCapIdx[1][2] = nCaps
		else
			imageCapIdx[i][1] = imageCapIdx[i-1][2] + 1
			imageCapIdx[i][2] = imageCapIdx[i-1][2] + nCaps
		end
	end
	return imageCapIdx, maxLength
end

function caputils.convertPath(data)
	local maxLength = -1
	for _, img in pairs(data) do
		maxLength = math.max(maxLength, #img['file_path']+1)
	end
	local imagePath = torch.CharTensor(#data, maxLength):zero()
	for i, img in pairs(data) do
		ffi.copy(imagePath[i]:data(), img['file_path'])
	end
	return imagePath
end

local function fs(t,a,b)
	if t[b] < t[a] then
		return true
	elseif t[b] == t[a] then
		return a < b
	else
		return false
	end
end

function caputils.buildVocab(data)
	local counts = {}
	for _, img in pairs(data) do
		for _, txt in pairs(img['processed_tokens']) do
			
			for _, w in pairs(txt) do
				if not counts[w] then
					counts[w] = 1
				else
					counts[w] = counts[w] + 1
				end
			end
		end
	end

	-- number the vocabulary from 1 to #vocab
	-- according to the order of counts from large to small
	local vocab = {}
	local num = 0
	for k, v in utils.spairs(counts, fs) do
		local word = {}
		num = num + 1
		word['id'] = num
		word['count'] = v
		vocab[k] = word
	end
	num = num + 1
	vocab['UNK'] = {id=num, count=0}

	local devocab = {}
	for k, v in pairs(vocab) do
		devocab[v['id']] = k
	end
	return vocab, devocab
end

function caputils.convertCaption(data, vocab, imageCapIdx, maxLength)
	local imageCaptions = torch.LongTensor(imageCapIdx[imageCapIdx:size(1)][2], maxLength)

	for i, img in pairs(data) do
		local nCaps = #(img['processed_tokens'])
		assert(nCaps>0, 'error: some image has no captions')
		for j, txt in pairs(img['processed_tokens']) do
			assert(#txt>0, 'error: some caption had no words')
			for k, w in pairs(txt) do
				if not vocab[w] then
					imageCaptions[imageCapIdx[i][1] + j - 1][k] = vocab['UNK']['id']
				else
					imageCaptions[imageCapIdx[i][1] + j - 1][k] = vocab[w]['id']
				end
			end
		end
	end
	return imageCaptions
end

return caputils