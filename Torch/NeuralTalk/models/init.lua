--[[
--
--  Generic model creating code.
--
--]]
require 'nn'
require 'cunn'
require 'cudnn'

require 'models.featureToSeq'
require 'models.featureToSeqSkip'
require 'models.seqCrossEntropyCriterion'

local modelutils = require 'utils.modelutils'

local M = {}

function M.setup(opt, vocabSize, checkpoint)
	local cnn, nFeatures
	if checkpoint then
		local modelPath = paths.concat(opt.resume, checkpoint.cnnModelFile)
		assert(paths.filep(modelPath), 'Saved cnn model not found: ' .. modelPath)
		print('<model init> => Resuming cnn model from ' .. modelPath)
		cnn = torch.load(modelPath)
	elseif opt.retrain ~= 'none' then
		assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
		print('<model init> => Loading cnn model from file: ' .. opt.retrain)
		cnn = torch.load(opt.retrain)
	elseif opt.cnnCaffe ~= 'none' then
		assert(paths.filep(opt.cnnCaffe), 'File not found: ' .. opt.cnnCaffe)
		assert(paths.filep(opt.cnnProto), 'File not found: ' .. opt.cnnProto)
		print('<model init> => Loading caffe model from file: ' .. opt.cnnCaffe)
		cnn = modelutils.buildCNN(opt)
		--print('<model init> => Saving caffe model to t7 file')
		--torch.save(opt.cnnCaffe .. '.t7', cnn)
	else
		--print('<model init> => Creating cnn model from file: models/' .. opt.cnnType .. '.lua')
		--cnn, nFeatures = require('models.' .. opt.cnnType)(opt)
		error('<model init> invalid cnn model')
	end

	if torch.type(model) == 'nn.DataParallelTable' then
		cnn = cnn:get(1)
	end

	
	if not checkpoint then
		-- for resetting the cnn last layer when fine-tuning on a different Dataset
		if opt.resetCNNlastlayer then
			print('<model init>  => Replacing cnn last layer with ' .. opt.encodingSize .. '-way FC layer')

			local orig = cnn:get(#cnn.modules)
			assert(torch.type(orig) == 'nn.Linear', 'expected last layer to be fully connected')

			nFeatures = orig.weight:size(2)
			cnn:remove(#cnn.modules)
			orig = cnn:get(#cnn.modules)
			if torch.type(orig) ~= 'nn.Dropout' then
				cnn:add(nn.Dropout(opt.cnnFCdropout))
			end
		else
			-- if cnn model already ends at feature layer
			nFeatures = opt.cnnFeatures
		end
	end

	local feature2seq, feature2seqTMP
	if checkpoint then
		local modelPath = paths.concat(opt.resume, checkpoint.seqModelFile)
		assert(paths.filep(modelPath), 'Saved seq model not found: ' .. modelPath)
		print('<model init> => Resuming seq model from ' .. modelPath)
		feature2seqTMP = torch.load(modelPath)
		feature2seq = modelutils.createFeature2seq(feature2seqTMP)
		modelutils.preproFeature2seq(feature2seq, feature2seqTMP)
	elseif opt.retrainlstm ~= 'none' then
		assert(paths.filep(opt.retrainlstm), 'File not found: ' .. opt.retrainlstm)
		print('<model init> => Loading lstm model from file: ' .. opt.retrainlstm)
		feature2seqTMP = torch.load(opt.retrainlstm)
		feature2seq = modelutils.createFeature2seq(feature2seqTMP)
		modelutils.preproFeature2seq(feature2seq, feature2seqTMP)
	elseif not opt.skipFlag then
		print('<model init> => Creating feature2seq model')
		feature2seq = nn.FeatureToSeq(opt, nFeatures, vocabSize)
	else
		print('<model init> => Creating feature2seq skipping model')
		feature2seq = nn.FeatureToSeqSkip(opt, nFeatures, vocabSize)
	end
	
	local criterion = nn.SeqCrossEntropyCriterion()

	if opt.nGPU < 1 then
		cnn:float()
		feature2seq:float()
		criterion:float()
	-- Wrap the model with DataParallelTable, if using more than one GPU
	elseif opt.nGPU > 1 then
		local gpus = torch.range(1, opt.nGPU):totable()
		local fastest, benchmark = cudnn.fastest, cudnn.benchmark

		local dpt = nn.DataParallelTable(1, true, true)
			:add(cnn, gpus)
			:threads(function()
				local cudnn = require 'cudnn'
				cudnn.fastest, cudnn.benchmark = fastest, benchmark
			end)
		dpt.gradInput = nil
		-- Set the CUDNN flags
		cudnn.fastest = true
		cudnn.benchmark = true
		cnn = dpt:cuda()
		feature2seq:cuda()
		criterion:cuda()
	else
		-- Set the CUDNN flags
		cudnn.fastest = true
		cudnn.benchmark = true
		cnn:cuda()
		feature2seq:cuda()
		criterion:cuda()
	end

	local model = {}
	model.cnn = cnn
	model.feature2seq = feature2seq

	collectgarbage()
	
	return model, criterion
end

return M
