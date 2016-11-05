require 'loadcaffe'

local netutils = {}

function netutils.convInit(model, name)
	for k,v in pairs(model:findModules(name)) do
		local n = v.kW*v.kH*v.nOutputPlane
		v.weight:normal(0,math.sqrt(2/n))
		if cudnn.version >= 4000 then
			v.bias = nil
			v.gradBias = nil
		else
			v.bias:zero()
		end
	end
end

function netutils.linearInit(model)
	for k,v in pairs(model:findModules('nn.Linear')) do
		local n = v.weight:size(2)+v.weight:size(1)
		v.weight:normal(0,math.sqrt(2/n))
		v.bias:zero()
	end
end

function netutils.build_cnn(opt)
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

return netutils