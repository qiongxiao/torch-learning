require 'torch'
require 'nn'
require 'image'

local layer, parent = torch.class('nn.BatchFlip', 'nn.Module')

function layer:__init()
	parent.__init(self)
	self.train = true
end

-- code from https://github.com/szagoruyko/cifar.torch/blob/master/train.lua
function layer:updateOutput( input )
	if self.train then
		self.output:resizeAs(input):copy(input)
		local batch_size = input:size(1)
		self.flip_mask = torch.randperm(bs):le(bs/2)
		for i = 1, batch_size do
			if flip_mask[i] == 1 then image.hflip(self.output[i], input[i]) end
		end
	else
		self.output:set(input)
	end
	return self.output
end

function layer:updateGradInput( input, gradOutput )
	if self.train then
		self.output:resizeAs(input):copy(input)
		local batch_size = input:size(1)
		for i = 1, batch_size do
			if flip_mask[i] == 1 then image.hflip(self.output[i], input[i]) end
		end
	else
		self.output:set(input)
	end
	return self.output
end