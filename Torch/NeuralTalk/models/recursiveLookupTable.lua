--[[
--
--  code from 'https://github.com/karpathy/neuraltalk2/blob/master/misc/LanguageModel.lua'
--
--]]

local nn = require 'nn'

local layer, parent = torch.class('nn.RLookupTable', 'nn.Module')

function layer:__init(nIndex, size, rLength, skipFlag)
	self.nIndex = nIndex
	self.outputSize = size
	self.cell = nn.LookupTable(nIndex, size)
	self.rLength = rLength
	skipFlag = skipFlag or false
	self.skipFlag = skipFlag
end

function layer:createSlices()
	self.slices = { self.cell }
	for t = 2, self.rLength do
		self.slices[t] = self.cell:clone('weight', 'bias', 'gradWeight', 'gradBias')
	end
end

function layer:getModule()
	return self.cell
end

function layer:parameters()
	return self.cell:parameters()
	-- ################ ?? destroy weight sharing?? ################
end

function layer:training()
	if self.slices == nil then self:createSlices() end -- create these lazily if needed
	for k,v in pairs(self.slices) do v:training() end
end

function layer:evaluate()
	if self.slices == nil then self:createSlices() end -- create these lazily if needed
	for k,v in pairs(self.slices) do v:evaluate() end
end

function layer:updateOutput(input)
	if self.slices == nil then self:createSlices() end -- lazily create clones on first forward pass
	assert(input:size(1) == self.rLength)
	local batchsize = input:size(2)
	self.output:resize(self.rLength, batchsize, self.outputSize)

	self.inputs = {}
	self.tmax = 0 -- we will keep track of max sequence length encountered in the data for efficiency
	local can_skip = false
	for t = 1, self.rLength do
		local it = input[t]:clone()
		if torch.sum(it) == 0 then
			can_skip = true
		end
		it[torch.eq(it, 0)] = 1
		if not (self.skipFlag and can_skip) then
			self.inputs[t] = it
			self.output[t] = self.slices[t]:forward(it)
			self.tmax = t
		else
			self.output[t]:zero()
		end
	end

	return self.output
end

function layer:updateGradInput(input, gradOutput)
	for t = self.tmax, 1, -1 do
		local it = self.inputs[t]
		self.slices[t]:backward(it, gradOutput[t])
	end
	if self.tmax < self.rLength then
		for t = self.tmax + 1, self.rLength do
			self.gradInput[t]:zero()
		end
	end
	return self.gradInput
end