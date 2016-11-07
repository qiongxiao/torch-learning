require 'loadcaffe'

local modelutils = {}

function modelutils.createFeature2seq(feature2seqTMP)
	local feature2seq
	local netType = torch.type(feature2seqTMP)
	if netType == 'nn.FeatureToSeqSkip' then
		feature2seq = nn.FeatureToSeqSkip(feature2seqTMP.opt, feature2seqTMP.nFeatures, feature2seqTMP.vocabSize)
	else
		feature2seq = nn.FeatureToSeq(feature2seqTMP.opt, feature2seqTMP.nFeatures, feature2seqTMP.vocabSize)
	end
	if feature2seqTMP.opt.nGPU > 0 then
		feature2seqTMP:cuda()
		feature2seq:cuda()
	end
	return feature2seq
end

function modelutils.cloneFeature2seq(feature2seq, feature2seqTMP)
	local moduleList = feature2seq:getModulesList()
	local origModuleList = feature2seqTMP:getModulesList()
	for k, v in pairs(moduleList) do v:share(origModuleList[k], 'weight', 'bias') end
end

function modelutils.preproFeature2seq(feature2seq, feature2seqTMP)
	feature2seq:createSlices()
	modelutils.cloneFeature2seq(feature2seq, feature2seqTMP)
	feature2seq:shareSlices()
end

function modelutils.buildCNN(opt)
	local cnn_raw = loadcaffe.load(opt.cnnProto, opt.cnnCaffe, opt.backendCaffe)

	-- copy over the first layer_num layers of the CNN
	local cnn_part = nn.Sequential()
	for i = 1,  opt.cnnCaffelayernum do
		local layer = cnn_raw:get(i)

		if i == 1 then
			-- convert kernels in first conv layer into RGB format instead of BGR,
			-- which is the order in which it was trained in Caffe
			local w = layer.weight:clone()
			-- swap weights to R and B channels
			print('<model init> => converting caffe cnn model first layer conv filters from BGR to RGB...')
			layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
			layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
		end

		cnn_part:add(layer)
	end
	return cnn_part
end

return modelutils