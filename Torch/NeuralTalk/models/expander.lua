--[[
--
--  code from 'https://github.com/karpathy/neuraltalk2/blob/master/misc/net_utils.lua'
--
--]]

require 'nn'

local layer, parent = torch.class('nn.Expander', 'nn.Module')

function layer:__init(dupSize)
	parent.__init(self)
	self.dupSize = dupSize
end

function layer:updateOutput(input)
	if self.dupSize == 1 then
		self.output = input
		return output
	end
	local outputSize = torch.LongStorage(input:size()):copy(input:size())
	local smallSize = torch.LongStorage(input:size()):copy(input:size())
	outputSize[1] = outputSize[1] * self.dupSize
	smallSize[1] = self.dupSize
	self.output:resize(outputSize)
	for k = 1, input:size(1) do
		local j = (k - 1) * self.dupSize
		self.output[{{j+1, j+self.dupSize}}] = input[{{k}}]:expand(smallSize)
	end
	return self.output
end

function layer:updateGradInput(input, gradOutput)
	if self.dupSize == 1 then
		self.gradInput = gradOutput
		return self.gradInput
	end
	self.gradInput:resizeAs(input)
	for k = 1, input:size(1) do
		local j = (k - 1) * self.dupSize
		self.gradInput[k] = torch.sum(gradOutput[{{j+1, j+self.dupSize}}], 1)
	end
	return self.gradInput
end