require 'torch'
require 'nn'

local LN, parent = torch.class('nn.LeNetModel', 'nn.Module')

function LN:__init()
	parent.__init(self)
	self.net = nn.Sequential()
	-- stage CONV - 1
	self.net:add(nn.SpatialConvolution(1, 32, 5, 5))
	self.net:add(nn.ReLU())
	self.net:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
	-- stage CONV - 2
	self.net:add(nn.SpatialConvolution(32, 64, 5, 5))
	self.net:add(nn.ReLU())
	self.net:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
	-- stage FC - 1
	self.net:add(nn.Reshape(64*4*4))
	self.net:add(nn.Linear(64*4*4, 500))
	self.net:add(nn.ReLU())
	-- stage FC - 2
	self.net:add(nn.Linear(500, 84))
	self.net:add(nn.ReLU())
	self.net:add(nn.Linear(84, 10))

	self.net = require('weight-init')(self.net, 'xavier')
end

function LN:updateOutput( input )
	return self.net:forward(input)
end

function LN:backward( input, gradOutput, scale )
	return self.net:backward(input, gradOutput, scale)
end

function LN:parameters()
	return self.net:parameters()
end

function LN:training()
	self.net:training()
	parent.training(self)
end

function LN:evaluate()
	self.net:evaluate()
	parent.evaluate(self)
end
