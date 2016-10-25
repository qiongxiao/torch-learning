local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local BatchNorm = nn.BatchNormalization

local function createModel(opt)
	local function ConvBNReLU( nInputPlane, nOutputPlane )
		return nn.Sequential():add(Convolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
			:add(SBatchNorm(nOutputPlane))
			:add(ReLU(true))
	end

	local size, nClasses
	if opt.dataset == 'cifar10' opt.dataset == 'cifar100' then
		size = 1
		nClasses = (opt.dataset == 'cifar10') and 10 or 100
	elseif opt.dataset == 'imagenet' then
		size = 7
		nClasses = 1000
	elseif opt.dataset == 'mnist' then
		error('invalid combination of mnist dataset and VGG net')
	else
		error('invalid dataset: ' .. opt.dataset)
	end

	local model = nn.Sequential()

	model:add(ConvBNReLU(iChanel, 64))
	if opt.conv_dropout > 0 then
		model:add(nn.Dropout(opt.convDropout))
	end
	model:add(ConvBNReLU(64, 64))
	model:add(Max(2, 2, 2, 2, 0, 0):ceil())

	-- stage CONV 2
	model:add(ConvBNReLU(64, 128))
	if opt.convDropout > 0 then
		model:add(nn.Dropout(opt.convDropout))
	end
	model:add(ConvBNReLU(128, 128))
	model:add(Max(2, 2, 2, 2, 0, 0):ceil())

	-- stage CONV 3
	model:add(ConvBNReLU(128, 256))
	if opt.convDropout > 0 then
		model:add(nn.Dropout(opt.convDropout))
	end
	model:add(ConvBNReLU(256, 256))
	if opt.convDropout > 0 then
		model:add(nn.Dropout(opt.convDropout))
	end
	model:add(ConvBNReLU(256, 256))
	model:add(Max(2, 2, 2, 2, 0, 0):ceil())

	-- stage CONV 4
	model:add(ConvBNReLU(256, 512))
	if opt.convDropout > 0 then
		model:add(nn.Dropout(opt.convDropout))
	end
	model:add(ConvBNReLU(512, 512))
	if opt.convDropout > 0 then
		model:add(nn.Dropout(opt.convDropout))
	end
	model:add(ConvBNReLU(512, 512))
	model:add(Max(2, 2, 2, 2, 0, 0):ceil())

	-- stage CONV 5
	model:add(ConvBNReLU(512, 512))
	if opt.convDropout > 0 then
		model:add(nn.Dropout(opt.convDropout))
	end
	model:add(ConvBNReLU(512, 512))
	if opt.convDropout > 0 then
		model:add(nn.Dropout(opt.convDropout))
	end
	model:add(ConvBNReLU(512, 512))
	model:add(Max(2, 2, 2, 2, 0, 0):ceil())

	model:add(nn.View(512 * size))

	if opt.dropout > 0 then
		model:add(nn.Dropout(opt.dropout))
	end
	model:add(nn.Linear(512, 512))
	model:add(nn.BatchNormalization(512))
	model:add(nn.ReLU())

	if opt.dropout > 0 then
		model:add(nn.Dropout(opt.dropout))
	end
	model:add(nn.Linear(512, nClasses))

	local function ConvInit(name)
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
	local function BNInit(name)
		for k,v in pairs(model:findModules(name)) do
			v.weight:fill(1)
			v.bias:zero()
		end
	end

	ConvInit('cudnn.SpatialConvolution')
	ConvInit('nn.SpatialConvolution')
	BNInit('fbnn.SpatialBatchNormalization')
	BNInit('cudnn.SpatialBatchNormalization')
	BNInit('nn.SpatialBatchNormalization')
	for k,v in pairs(model:findModules('nn.Linear')) do
		v.bias:zero()
	end
	model:cuda()
	if opt.cudnn == 'deterministic' then
		model:apply(function(m)
			if m.setMode then m:setMode(1,1,1) end
		end)
	end

	model:get(1).gradInput = nil

	return model
end

return createModel