require 'torch'
require 'nn'
require 'BatchFlip'
require 'VGGNet'

local utils = require 'utils'

local VN, parent = torch.class('nn.NetModel', 'nn.Module')

function VN:__init(kwargs)
	parent.__init(self)
	local batch_size = utils.get_kwarg(kwargs, 'batch_size')
	self.data_flip = utils.get_kwarg(kwargs, 'data_flip')
	self.conv_dropout = utils.get_kwarg(kwargs, 'conv_dropout')
	self.spatial_batchnorm = utils.get_kwarg(kwargs, 'spatial_batchnorm')
	self.dropout = utils.get_kwarg(kwargs, 'dropout')
	self.batchnorm = utils.get_kwarg(kwargs, 'batchnorm')
	self.model_type = utils.get_kwarg(kwargs, 'model_type')

	self.net = nn.Sequential()

	if data_flip == 1 then
		self.net:add(nn.BatchFlip())
	end

	local cnn
	if self.model_type == 'VGG' then
		cnn = nn.VGGNet(kwargs)
	end

	self.net:add(cnn)
end

function VN:updateOutput( input )
	return self.net:forward(input)
end

function VN:backward( input, gradOutput, scale )
	return self.net:backward(input, gradOutput, scale)
end

function VN:parameters()
	return self.net:parameters()
end

function VN:training()
	self.net:training()
	parent.training(self)
end

function VN:evaluate()
	self.net:evaluate()
	parent.evaluate(self)
end
