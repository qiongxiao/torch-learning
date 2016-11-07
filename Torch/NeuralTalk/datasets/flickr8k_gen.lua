--[[
--
--  flickr8k dataset preparation
--
--]]

require 'torch'
local paths = require 'paths'

local utils = require 'utils.utils'
local caputils = require 'utils.caputils'

local URL1 = 'http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_Dataset.zip'
local URL2 = 'http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_text.zip'

local M = {}

local function createSplit(path)
	local out = {}
	for line in io.lines(path) do
		out[line] = 0
	end
	return out
end

local function createStructure(path, dictionary)
	local out = {}
	local tmp = {}
	for line in io.lines(path) do
		local sp = string.find(line, '#')
		local file_path = string.sub(line, 1, sp-1)
		local annot = string.sub(line, sp+3, -1)

		local res = dictionary[file_path]
		if res then
			local img = tmp[file_path]
			if not img then
				img = {}
				img['file_path'] = file_path
				img['captions'] = {}
				table.insert(img['captions'], annot)
				tmp[file_path] = img
			else
				table.insert(img['captions'], annot)
			end
		end
	end
	for _, img in pairs(tmp) do
		table.insert(out, img)
	end
	collectgarbage()
	return out
end



function M.exec(opt, cacheFile)
	local dir = paths.concat(opt.data)
	if not paths.dirp(dir) then
		print('image directory not found: ' .. dir)
		print("<data init> => Downloading flickr8k image dataset from " .. URL1)
		local ok = os.execute('wget ' .. URL1)
		assert(ok == true or ok == 0, 'error downloading flickr8k')
		os.execute('mkdir gen/Flickr8k_Dataset')
		ok = os.execute('unzip Flickr8k_Dataset.zip -d gen/')
		assert(ok == true or ok == 0, 'error unzip flickr8k image')
		opt.data = 'gen/Flicker8k_Dataset'
	end

	print("<data init> => Downloading flickr8k text dataset from " .. URL2)
	local ok = os.execute('wget ' .. URL2)
	assert(ok == true or ok == 0, 'error downloading flickr8k')
	os.execute('mkdir gen/Flickr8k_text')
	ok = os.execute('unzip Flickr8k_text.zip -d gen/Flickr8k_text/')
	assert(ok == true or ok == 0, 'error unzip flickr8k text')

	print("<data init> | preprocessing train data")
	local dic = createSplit('gen/Flickr8k_text/Flickr_8k.trainImages.txt')
	local out = createStructure('gen/Flickr8k_text/Flickr8k.token.txt', dic)
	print("<data init> | preprocessing train data captions stage 1")
	local trainImageCapIdx, maxLength = caputils.preproCaptions(out)
	print("<data init> | preprocessing train data image path")
	local trainImagePath = caputils.convertPath(out)
	print("<data init> | building vocabulary")
	local vocab, devocab = caputils.buildVocab(out)
	print("<data init> | preprocessing train data captions stage 2")
	local trainImageCaptions  = caputils.convertCaption(out, vocab, trainImageCapIdx, maxLength)

	print("<data init> | preprocessing val data")
	dic = createSplit('gen/Flickr8k_text/Flickr_8k.devImages.txt')
	out = createStructure('gen/Flickr8k_text/Flickr8k.token.txt', dic)
	print("<data init> | preprocessing val data captions stage 1")
	local valImageCapIdx, maxLength = caputils.preproCaptions(out)
	print("<data init> | preprocessing val data image path")
	local valImagePath = caputils.convertPath(out)
	print("<data init> | preprocessing val data captions stage 2")
	local valImageCaptions  = caputils.convertCaption(out, vocab, valImageCapIdx, maxLength)

	print("<data init> | preprocessing test data")
	dic = createSplit('gen/Flickr8k_text/Flickr_8k.testImages.txt')
	out = createStructure('gen/Flickr8k_text/Flickr8k.token.txt', dic)
	print("<data init> | preprocessing test data captions stage 1")
	local testImageCapIdx, maxLength = caputils.preproCaptions(out)
	print("<data init> | preprocessing test data image path")
	local testImagePath = caputils.convertPath(out)
	print("<data init> | preprocessing test data captions stage 2")
	local testImageCaptions = caputils.convertCaption(out, vocab, testImageCapIdx, maxLength)

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
		test = {
			imagePath = testImagePath,
			imageCaptions = testImageCaptions,
			imageCapIdx = testImageCapIdx,
		},
	}
	print("<data init> | saving flickr8k dataset to " .. cacheFile)
	torch.save(cacheFile, info)
	collectgarbage()
	return info
end

return M
