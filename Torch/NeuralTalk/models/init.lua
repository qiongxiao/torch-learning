--[[
--
--  code from https://github.com/facebook/fb.resnet.torch/blob/master/models/init.lua
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--
--]]
require 'nn'
require 'cunn'
require 'cudnn'

require 'models.featureToSeq'
require 'models.seqCrossEntropyCriterion'

local M = {}

function M.setup(opt, vocabSize, checkpoint)
	local cnn, nFeatures
	if checkpoint and checkpoint.cnnModelFile then
		local modelPath = paths.concat(opt.resume, checkpoint.cnnModelFile)
		assert(paths.filep(modelPath), 'Saved cnn model not found: ' .. modelPath)
		print('<model init> => Resuming cnn model from ' .. modelPath)
		cnn = torch.load(modelPath)
	elseif opt.retrain ~= 'none' then
		assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
		print('<model init> => Loading cnn model from file: ' .. opt.retrain)
		cnn = torch.load(opt.retrain)
	else
		print('<model init> => Creating cnn model from file: models/' .. opt.cnnType .. '.lua')
		cnn, nFeatures = require('models.' .. opt.cnnType)(opt)
	end

	if torch.type(model) == 'nn.DataParallelTable' then
		cnn = cnn:get(1)
	end

	-- For resetting the cnn last layer when fine-tuning on a different Dataset
	if opt.resetCNNlastlayer and not checkpoint then
		print('<model init>  => Replacing cnn last layer with ' .. opt.encodingSize .. '-way FC layer')

		local orig = cnn:get(#cnn.modules)
		assert(torch.type(orig) == 'nn.Linear', 'expected last layer to be fully connected')

		nFeatures = orig.weight:size(2)
		cnn:remove(#cnn.modules)
	end

	local feature2seq
	if checkpoint and checkpoint.seqModelFile then
		local modelPath = paths.concat(opt.resume, checkpoint.seqModelFile)
		assert(paths.filep(modelPath), 'Saved seq model not found: ' .. modelPath)
		print('<model init> => Resuming seq model from ' .. modelPath)
		feature2seq = torch.load(modelPath)
	else
		print('<model init> => Creating feature2seq model')
		feature2seq = nn.FeatureToSeq(opt, nFeatures, vocabSize)
	end
	
	cnn:cuda()
	feature2seq:cuda()

	-- Set the CUDNN flags
	cudnn.fastest = true
	cudnn.benchmark = true

	local criterion = nn.SeqCrossEntropyCriterion():cuda()
	return cnn, feature2seq, criterion
end

return M
