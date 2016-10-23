require 'torch'
require 'nn'

local utils = require 'utils'

local VN, parent = torch.class('nn.VGGNet', 'nn.Module')

function VN:__init(kwargs, output_dim)
	parent.__init(self)
	self.conv_dropout = utils.get_kwarg(kwargs, 'conv_dropout')
	self.spatial_batchnorm = utils.get_kwarg(kwargs, 'spatial_batchnorm')
	self.dropout = utils.get_kwarg(kwargs, 'dropout')
	self.batchnorm = utils.get_kwarg(kwargs, 'batchnorm')

	self.net = nn.Sequential()
	-- code from 'https://github.com/szagoruyko/cifar.torch/blob/master/models/vgg_bn_drop.lua'
	local function ConvBNReLU( nInputPlane, nOutputPlane )
		self.net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
		if self.spatial_batchnorm == 1 then
			self.net:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
		end
		self.net:add(nn.ReLU())
	end

	-- stage CONV 1
	ConvBNReLU(3, 64)
	if self.conv_dropout > 0 then
		self.net:add(nn.Dropout(self.conv_dropout))
	end
	ConvBNReLU(64, 64)
	self.net:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())

	-- stage CONV 2
	ConvBNReLU(64, 128)
	if self.conv_dropout > 0 then
		self.net:add(nn.Dropout(self.conv_dropout))
	end
	ConvBNReLU(128, 128)
	self.net:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())

	-- stage CONV 3
	ConvBNReLU(128, 256)
	if self.conv_dropout > 0 then
		self.net:add(nn.Dropout(self.conv_dropout))
	end
	ConvBNReLU(256, 256)
	if self.conv_dropout > 0 then
		self.net:add(nn.Dropout(self.conv_dropout))
	end
	ConvBNReLU(256, 256)
	self.net:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())

	-- stage CONV 4
	ConvBNReLU(256, 512)
	if self.conv_dropout > 0 then
		self.net:add(nn.Dropout(self.conv_dropout))
	end
	ConvBNReLU(512, 512)
	if self.conv_dropout > 0 then
		self.net:add(nn.Dropout(self.conv_dropout))
	end
	ConvBNReLU(512, 512)
	self.net:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())

	-- stage CONV 5
	ConvBNReLU(512, 512)
	if self.conv_dropout > 0 then
		self.net:add(nn.Dropout(self.conv_dropout))
	end
	ConvBNReLU(512, 512)
	if self.conv_dropout > 0 then
		self.net:add(nn.Dropout(self.conv_dropout))
	end
	ConvBNReLU(512, 512)
	self.net:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())

	-- 32/2/2/2/2/2 = 1
	self.net:add(nn.View(512))

	-- stage FC 1
	if self.dropout > 0 then
		self.net:add(nn.Dropout(self.dropout))
	end
	self.net:add(nn.Linear(512, 512))
	if self.batchnorm == 1 then
		self.net:add(nn.BatchNormalization(512))
	end
	self.net:add(nn.ReLU())
	-- stage FC 2
	if self.dropout > 0 then
		self.net:add(nn.Dropout(self.dropout))
	end
	self.net:add(nn.Linear(512, output_dim))

	self.net = require('weight-init')(self.net, 'xavier')
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
