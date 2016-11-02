--[[
--
--  code from "https://github.com/karpathy/neuraltalk2/blob/master/coco/coco_preprocess.ipynb"
--
--]]

local utils = require 'utils/utils'
local sys = require 'sys'
local ffi = require 'ffi'
local paths = require 'paths'

local URL = 'http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip'

local M = {}

local function mergeAnnot(path)
	local itoa = {}
	local out = {}

	local data = utils.readJson(path)
	local annots = data['annotations']
	local imgs = data['images']
	for _, v in pairs(annots) do
		local imgid = v[image_id]
		if not itoa[imgid] then
			itoa[imgid] = {}
		end
		table.insert(itoa[imgid], v['caption'])
	end
	for i, img in pairs(imgs) do
		local imgid = img['id']
		local imgNew = {}
		imgNew['file_path'] = img['file_name']
		imgNew['id'] = imgid
		imgNew['captions'] = itoa[imgid]
		table.insert(out, imgNew)
	end
	return out
end

local function preproCaptions(data)
	for i, img in pairs(data) do
		img['processed_tokens'] = {}
		for _, s in pairs(img['captions']) do
			local tokens = {}
			local txt = string.gsub(s, "%p", "")
			for token in string.gmatch(txt, "[^%s]+") do
				table.insert(tokens, token)
			end
			table.insert(img['processed_tokens'], tokens)
		end
	end
end

local function convertPath(data)
	local maxLength = -1
	for _, img in pairs(data) do
		maxLength = math.max(maxLength, #img['file_path'])
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

-- code from http://stackoverflow.com/questions/15706270/sort-a-table-in-lua
function spairs(t, order)
    local keys = {}
    for k in pairs(t) do keys[#keys+1] = k end

    if order then
        table.sort(keys, function(a,b) return order(t, a, b) end)
    else
        table.sort(keys)
    end

    local i = 0
    return function()
        i = i + 1
        if keys[i] then
            return keys[i], t[keys[i]]
        end
    end
end

local function buildVocab(data)
	local maxLength = -1
	local counts = {}
	for _, img in pairs(data) do
		for _, txt in pairs(img['processed_tokens']) do
			maxLength = math.max(maxLength, #txt)
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
	for k, v in spairs(counts, fs) do
		local word = {}
		word['id'] = #vocab + 1
		word['count'] = v
		vocab[k] = word
	end
	local num = #vocab + 1
	vocab['UNK'] = {id=num, count=0}

	local devocab = {}
	for k, v in pairs(vocab) do
		devocab[v['id']] = k
	end

	return vocab, devocab, maxLength
end

local function convertCaption(data, vocab, maxLength)
	local imageCaptions
	local imageCapIdx = torch.LongTensor(#data, 2):zero()
	for i, img in pairs(data) do
		local nCaps = #(img['processed_tokens'])
		assert(nCaps>0, 'error: some image has no captions')
		local captions = torch.LongTensor(nCaps, maxLength)
		for j, txt in pairs(img['processed_tokens']) do
			assert(#txt>0, 'error: some caption had no words')
			for k, w in pairs(txt) do
				if not vocab[w] then
					captions[j][k] = vocab['UNK']['id']
				else
					captions[j][k] = vocab[w]['id']
				end
			end
		end
		if i == 1 then
			imageCaptions = captions
			imageCapIdx[1][1] = 1
			imageCapIdx[1][2] = nCaps
		else
			imageCaptions = torch.cat(imageCaptions, captions, 1)
			imageCapIdx[i][1] = imageCapIdx[i-1][2] + 1
			imageCapIdx[i][2] = imageCapIdx[i-1][2] + nCaps
		end
	end
	return imageCaptions, imageCapIdx
end

function M.exec(opt, cacheFile)
	print("<data init> => Downloading mscoco dataset from " .. URL)
	local ok = os.execute('wget ' .. URL)
	assert(ok == true or ok == 0, 'error downloading mscoco')
	ok = os.execute('unzip captions_train-val2014.zip gen/')
	assert(ok == true or ok == 0, 'error unzip mscoco')

	local trainDir = paths.concat(opt.data, 'train2014')
	local valDir = paths.concat(opt.data, 'val2014')
	assert(paths.dirp(trainDir), 'train directory not found: ' .. trainDir)
	assert(paths.dirp(valDir), 'val directory not found: ' .. valDir)

	print("<data init> | preprocessing data")
	local train = mergeAnnot('annotations/captions_train2014.json')
	local val = mergeAnnot('annotations/captions_val2014.json')

	preproCaptions(val)
	preproCaptions(train)

	local trainImagePath = convertPath(train)
	local valImagePath = convertPath(val)
	local vocab, devocab, maxLength = buildVocab(train)
	local trainImageCaptions, trainImageCapIdx = convertCaption(train, vocab, maxLength)
	local valImageCaptions, valImageCapIdx = convertCaption(val, vocab, maxLength)

	local info = {
		basedir = opt.data,
		vocab = vocab,
		devocab = devocab,
		train = {
			imagePath = trainImagePath,
			imageCaptions = trainImageCaptions,
			imageCapIdx = trainImageCapIdx,
		},
		val = {
			imagePath = valImagePath,
			imageCaptions = valImageCaptions,
			imageCapIdx = valImageCapIdx,
		},
	}
	print("<data init> | saving mscoco dataset to " .. cacheFile)
	torch.save(cacheFile, info)
	collectgarbage()
	return info
end

return M
