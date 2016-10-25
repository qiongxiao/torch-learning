local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling

local function createModel(opt)
	local function SRM(nInputPlane, nOutputPlane)
		return nn.Sequential():add(Convolution(nInputPlane, nOutputPlane, 5, 5))
			:add(ReLU)
			:add(Max(2, 2, 2, 2, 0, 0))
	end

	local model = nn.Sequential()

	model:add(SRM(32, 64))
	model:add(SRM(64, 64))
	local size, nClasses
	if opt.dataset == 'imagenet' then
		size = 53 -- ((224 - 4) / 2 - 4) / 2
		nClasses = 1000
	elseif opt.dataset == 'cifar10' or opt.dataset == 'cifar100' then
		size = 5 -- ((32 - 4) / 2 - 4) / 2
		nClasses = (opt.dataset == 'cifar10') and 10 or 100
	elseif opt.dataset == 'mnist'
		size = 4 -- ((28 - 4) / 2 - 4) / 2
		nClasses = 10
	else
		error('invalid dataset ' .. opt.dataset)
	end
	model:add(nn.Reshape(64 * size * size))
	model:add(nn.Dropout(opt.dropout))
	model:add(Linear(64 * size * size, 1024))
	model:add(ReLU())
	model:add(nn.Linear(1024, nClasses))

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

	ConvInit('cudnn.SpatialConvolution')
	ConvInit('nn.SpatialConvolution')

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