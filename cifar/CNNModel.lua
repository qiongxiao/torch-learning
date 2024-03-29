require 'torch'
require 'nn'

require 'BatchFlip'
require 'VGGNet'

local utils = require 'utils'

local CNN, parent = torch.class('nn.CNNModel', 'nn.Module')

function CNN:__init(kwargs, output_dim)
	parent.__init(self)
	self.data_flip = utils.get_kwarg(kwargs, 'data_flip')

	self.net = nn.Sequential()

	if data_flip == 1 then
		self.net:add(nn.BatchFlip())
	end

	local cnn = nn.VGGNet(kwargs, output_dim)
	self.net:add(cnn)
end

function CNN:updateOutput( input )
	return self.net:forward(input)
end

function CNN:backward( input, gradOutput, scale )
	return self.net:backward(input, gradOutput, scale)
end

function CNN:parameters()
	return self.net:parameters()
end

function CNN:training()
	self.net:training()
	parent.training(self)
end

function CNN:evaluate()
	self.net:evaluate()
	parent.evaluate(self)
end
