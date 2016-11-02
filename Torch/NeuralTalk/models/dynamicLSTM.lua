--[[
--
--  code from 'https://github.com/karpathy/neuraltalk2/blob/master/misc/LanguageModel.lua'
--
--]]

require 'nn'

local layer, parent = torch.class('nn.dynamicLSTM', 'nn.Module')

function layer:__init(cell, rLength, rDepth, skipFlag)
	parent.__init(self)
	self.inputSize = inputSize
	self.outputSize = outputSize
	self.rLength = rLength
	self.rDepth = rDepth
	self.cell = cell
	self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batchsize)
	assert(batchsize ~= nil, 'batch size for dynamic lstm must be provided')
	-- construct the initial state for the LSTM
	if not self.initState then self.initState = {} end -- lazy init
	for h = 1, self.rDepth * 2 do
		-- note, the init state MUST be zeros because we are using initState to init grads in backward call too
		if self.initState[h] then
			if self.initState[h]:size(1) ~= batchsize then
				self.initState[h]:resize(batchsize, self.rnn_size):zero() -- expand the memory
			end
		else
			self.initState[h] = torch.zeros(batchsize, self.rnn_size)
		end
	end
	self.numState = #self.initState
end

function layer:createSlices()
	self.slices = { self.cell }
	for t = 2, self.rLength do
		self.slices[t] = self.cell:clone('weight', 'bias', 'gradWeight', 'gradBias')
	end
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

--[[
	input:
		torch.Tensor of size "rLength * batchsize * inputSize"
	output:
		torch.Tensor of size "rLenght * batchsize * outputSize"
--]]
function layer:updateOutput(input)
	if self.slices == nil then self:createSlices() end -- lazily create clones on first forward pass
	assert(input:size(1) == self.rLength)
	local batchsize = input:size(2)
	self.output:resize(rLength, batchsize, self.outputSize)

	self._createInitState(batchsize)

	self.state{[0] = self.initState}
	self.inputs = {}

	for t = 1, self.rLength do
		self.inputs[t] = {input[t], unpack(self.state[t-1])}
		local out = self.slices[t]:forward(self.inputs[t])
		self.output[t] = out[self.numState+1]
		self.state[t] = {}
		for i = 1, self.numState do
			table.insert(self.state[t], out[i])
		end
	end

	return self.output
end

function layer:updateGradInput(input, gradOutput)
	local dState = {[self.rLength] = self.initState}
	for t = self.rLength, 1, -1 do
		local dout = {}
		for k = 1, #dState[t] do
			table.insert(dout, dState[t][k])
		end
		table.insert(dout, gradOutput[t])
		local dInputs = self.slices[t]:backward(self.inputs[t], dout)
		self.gradInput[t] = dInputs[1]
		dState[t-1] = {}
		for k = 2, self.numState+1 do
			table.insert(dState[t-1], dInputs[k])
		end
	end
	return self.gradInput
end