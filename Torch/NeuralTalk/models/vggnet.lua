--[[
--
--  The vggnet model definition
--
--]]
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

	local nClasses = opt.encodingSize
	local size
	if opt.dataset == 'mscoco'
		size = 8 --256/2^5
	else
		error('invalid dataset: ' .. opt.dataset)
	end

	local model = nn.Sequential()

	model:add(ConvBNReLU(3, 64))
	if opt.conv_dropout > 0 then
		model:add(nn.Dropout(opt.cnnCONVdropout))
	end
	model:add(ConvBNReLU(64, 64))
	model:add(Max(2, 2, 2, 2, 0, 0):ceil())

	-- stage CONV 2
	model:add(ConvBNReLU(64, 128))
	if opt.cnnCONVdropout > 0 then
		model:add(nn.Dropout(opt.cnnCONVdropout))
	end
	model:add(ConvBNReLU(128, 128))
	model:add(Max(2, 2, 2, 2, 0, 0):ceil())

	-- stage CONV 3
	model:add(ConvBNReLU(128, 256))
	if opt.cnnCONVdropout > 0 then
		model:add(nn.Dropout(opt.cnnCONVdropout))
	end
	model:add(ConvBNReLU(256, 256))
	if opt.cnnCONVdropout > 0 then
		model:add(nn.Dropout(opt.cnnCONVdropout))
	end
	model:add(ConvBNReLU(256, 256))
	model:add(Max(2, 2, 2, 2, 0, 0):ceil())

	-- stage CONV 4
	model:add(ConvBNReLU(256, 512))
	if opt.cnnCONVdropout > 0 then
		model:add(nn.Dropout(opt.cnnCONVdropout))
	end
	model:add(ConvBNReLU(512, 512))
	if opt.cnnCONVdropout > 0 then
		model:add(nn.Dropout(opt.cnnCONVdropout))
	end
	model:add(ConvBNReLU(512, 512))
	model:add(Max(2, 2, 2, 2, 0, 0):ceil())

	-- stage CONV 5
	model:add(ConvBNReLU(512, 512))
	if opt.cnnCONVdropout > 0 then
		model:add(nn.Dropout(opt.cnnCONVdropout))
	end
	model:add(ConvBNReLU(512, 512))
	if opt.cnnCONVdropout > 0 then
		model:add(nn.Dropout(opt.cnnCONVdropout))
	end
	model:add(ConvBNReLU(512, 512))
	model:add(Max(2, 2, 2, 2, 0, 0):ceil())

	model:add(nn.View(512 * size * size))

	if opt.cnnFCdropout > 0 then
		model:add(nn.Dropout(opt.cnnFCdropout))
	end
	model:add(nn.Linear(512 * size * size, 4096))
	model:add(nn.BatchNormalization(4096))
	model:add(nn.ReLU())

	if opt.cnnFCdropout > 0 then
		model:add(nn.Dropout(opt.cnnFCdropout))
	end
	model:add(nn.Linear(4096, nClasses))

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
