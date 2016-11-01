--[[
--
--  code from 'https://github.com/karpathy/neuraltalk2/blob/master/misc/LanguageModel.lua'
--
--]]

require 'nn'
local lstmCell = require 'models.lstmCell'

local layer, parent = torch.class('nn.LSTM', 'nn.Module')

function layer:__init(inputSize, outputSize, hidenStateSize, rLength, rDepth, dropout, skipFlag)
	parent.__init(self)
	self.inputSize = inputSize
	self.outputSize = outputSize
	self.rLength = rLength
	self.rDepth = rDepth
	skipFlag = skipFlag or false
	self.skipFlag = skipFlag
	self.cell = lstmCell(inputSize, outputSize, hidenStateSize, rDepth, dropout)
	self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batchsize)
	assert(batchsize ~= nil, 'batch size for lstm must be provided')
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
	self.output:resize(rLength, batchsize, self.outputSize)

	self._createInitState(batchsize)

	self.state{[0] = self.initState}
	self.inputs = {}
	self.tmax = 0 -- we will keep track of max sequence length encountered in the data for efficiency
	local can_skip = false
	for t = 1, self.rLength do
		if torch.sum(input[t]) == 0 then
			can_skip = true
		end
		if not (self.skipFlag and can_skip) then
			self.inputs[t] = {input[t], unpack(self.state[t-1])}
			local out = self.slices[t]:forward(self.inputs[t])
			self.output[t] = out[self.numState+1]
			self.state[t] = {}
			for i = 1, self.numState do
				table.insert(self.state[t], out[i])
			end
			self.tmax = t
		end
	end

	return self.output
end

function layer:updateGradInput(input, gradOutput)
	local dState = {[self.tmax] = self.initState}
	for t = self.tmax, 1, -1 do
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
	if self.tmax < self.rLength then
		for t = self.tmax + 1, self.rLength do
			self.gradInput[t]:zero()
		end
	end

	return self.gradInput
end