--[[
--
--  code from "https://github.com/karpathy/neuraltalk2/blob/master/coco/coco_preprocess.ipynb"
--
--]]

local sys = require 'sys'
local ffi = require 'ffi'
local paths = require 'paths'

local utils = require 'utils.utils'
local caputils = require 'datasets.caputils'

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

	local trainImageCapIdx, valMaxLength = caputils.preproCaptions(val)
	local valImageCapIdx, trainMaxLength = caputils.preproCaptions(train)

	local trainImagePath = caputils.convertPath(train)
	local valImagePath = caputils.convertPath(val)
	local vocab, devocab = caputils.buildVocab(train)
	local trainImageCaptions = caputils.convertCaption(train, vocab, trainImageCapIdx, maxLength)
	local valImageCaptions = caputils.convertCaption(val, vocab, valImageCapIdx, maxLength)

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
