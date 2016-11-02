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
require 'models.expander'
require 'models.seqCrossEntropyCriterion'

local M = {}

function M.setup(opt, vocabSize, checkpoint)
	local cnn 
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
		cnn = require('models.' .. opt.cnnType)(opt)
	end

	local expander = nn.Expander(opt.seqPerImg)


	local feature2seq
	if checkpoint and checkpoint.seqModelFile then
		local modelPath = paths.concat(opt.resume, checkpoint.seqModelFile)
		assert(paths.filep(modelPath), 'Saved seq model not found: ' .. modelPath)
		print('<model init> => Resuming seq model from ' .. modelPath)
		feature2seq = torch.load(modelPath)
	else
		print('<model init> => Creating cnn model from file: models/' .. opt.cnnType .. '.lua')
		feature2seq = nn.FeatureToSeq(opt, vocabSize)
	end
	
	
	cnn:cuda()
	expander:cuda()
	feature2seq:cuda()

	-- Set the CUDNN flags
	cudnn.fastest = true
	cudnn.benchmark = true

	local criterion = nn.seqCrossEntropyCriterion():cuda()
	return cnn, expander, feature2seq, criterion
end

return M
