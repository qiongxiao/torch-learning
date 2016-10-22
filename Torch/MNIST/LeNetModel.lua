require 'torch'
require 'nn'

local LN, parent = torch.class('nn.LeNetModel', 'nn.Module')

function LN:__init()
	parent.__init(self)
	self.net = nn.Sequential()
	-- stage CONV - 1
	self.net:add(nn.SpatialConvolution(1, 6, 5, 5))
	self.net:add(nn.Relu())
	self.net:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
	-- stage CONV - 2
	self.net:add(nn.SpatialConvolution(6, 16, 5, 5))
	self.net:add(nn.Relu())
	self.net:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
	-- stage FC - 1
	self.net:add(nn.Reshape(16*4*4))
	self.net:add(nn.Linear(16*4*4, 120))
	self.net:add(nn.Relu())
	-- stage FC - 2
	self.net:add(nn.Linear(120, 84))
	self.net:add(nn.Relu())
	self.net:add(nn.Linear(84, 10))

	self.net = require('weight-init')(self.net, 'xavier')
end

function LN:updateOutput( input )
	return self.net:forward(input)
end

function LN:backward( input, gradOutput )
	return self.net:backward(input, gradOutput)
end

function LN:Parameters()
	return self.net:Parameters()
end

function LM:training()
	self.net:training()
	parent.training(self)
end

function LM:evaluate()
	self.net:evaluate()
	parent.evaluate(self)
end