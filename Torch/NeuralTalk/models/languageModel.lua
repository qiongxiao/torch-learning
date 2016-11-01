--[[
--
--  code from 'https://github.com/karpathy/neuraltalk2/blob/master/misc/LanguageModel.lua'
--
--]]

require 'nn'
require 'models.lstm'
local lstmCell = require 'models.lstmCell'

local layer, parent = torch.class('nn.LanguageModel', 'nn.Module')

function layer:__init(opt, vocabSize)
	parent.__init(self)

	self.vocabSize = vocabSize
	self.inputEncodingSize = opt.inputEncodingSize
	self.rDepth = opt.rDepth
	self.seqLength = opt.seqLength
	self.hiddenStateSize = opt.hiddenStateSize

	self.lstmCell = lstmCell(self.inputEncodingSize, self.vocabSize+1, self.hiddenStateSize, self.rDepth, opt.dropout)
	self.lstm = nn.LSTM(self.lstmCell, self.seqLength+2, self.rDepth)

	self.lookupTable = nn.LookupTable(self.vocabSize+1, self.inputEncodingSize)

	self.inferenceMax = opt.inferenceMax
	self.temperature = opt.temperature
end

function layer:_createInitState(batchsize)
	assert(batchsize ~= nil, 'batch size for language model must be provided')
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

function layer:getModulesList()
	return {self.lstmCell, self.lookupTable}
end

function layer:parameters()
	local p1,g1 = self.lstmCell:parameters()
	local p2,g2 = self.lookupTable:parameters()

	local params = {}
	for k,v in pairs(p1) do table.insert(params, v) end
	for k,v in pairs(p2) do table.insert(params, v) end

	local grad_params = {}
	for k,v in pairs(g1) do table.insert(grad_params, v) end
	for k,v in pairs(g2) do table.insert(grad_params, v) end

	-- ################ ?? destroy weight sharing?? ################
	return params, grad_params
end

function layer:training()
	self.lstm:training()
	self.lookupTable:training()
end

function layer:evaluate()
	self.lstm:evaluate()
	self.lookupTable:evaluate()
end

--[[
	input is a table of:
		1.	torch.Tensor of size "batchsize * inputEncodingSize"
		2.	torch.LongTensor of size "batchsize * seqLength", elements 1..M
			where M = vocabSize
	output:
		torch.Tensor of size "batchsize * (seqLength+2) * (M+1)"
--]]
function layer:updateOutput(input)
	local imgs = input[1]
	local seq = input[2]

	assert(imgs:size(2) == self.inputEncodingSize)
	assert(seq:size(2) == self.seqLength)

	local batchsize = seq:size(1)
	self.lookupTableInput = torch.cat({torch.Tensor(batchsize, 1):fill(self.vocabSize+1), seq})
	local seqEncoded = self.lookupTable:forward(lookupTableInput)
	-- seqEncoded's size is "batchsize * (seqLength+1) * inputEncodingSize"
	self.lstmInput = torch.cat({imgs:view(batchsize, 1, self.inputEncodingSize), seqEncoded}, 2)
	-- lstmInput's size is "batchsize * (seqLength+2) * inputEncodingSize"
	self.lstmInput = self.lstmInput:transpose(1, 2):contiguous()
	-- lstmInput's size is "(seqLength+2) * batchsize * inputEncodingSize"
	self.output = self.lstm:forward(self.lstmInput)
	self.output = self.output:transpose(1, 2):contiguous()
	return self.output
end

function layer:updateGradInput(input, gradOutput)
	local dImgs
	local dlstmInput = self.lstm:backward(self.lstmInput, gradOutput:transpose(1, 2))
	local dlookupTableInput = self.lookupTable:backward(self.rLookupTableInput, dlstmInput:narrow(1, 2, self.seqLength+1):transpose(1, 2))
	dImgs = dlstmInput[1]:view(input[1]:size())
	self.gradInput = {dImgs, torch.Tensor()}
	return self.gradInput
end

--[[
	input:
		torch.Tensor of size "batchsize * inputEncodingSize"
	output:
		torch.Tensor of size "batchsize * seqLength"
--]]
function layer:inference(imgs)
	local inferenceMax = self.inferenceMax
	local temperature = self.temperature

	local batchsize = imgs:size(1)
	self:_createInitState(batchsize)
	local state = self.initState

	local output = torch.LongTensor(self.seqLength, batchsize):zero()
	local outputProbs = torch.FloatTensor(self.seqLength, batchsize)
	local logprobs

	for t = 1, self.seqLength+2 do

		local xt, it, inferenceProbs

		if t == 1 then
			xt = imgs
		elseif t == 2 then
			it = torch.LongTensor(batchsize):fill(self.vocabSize+1)
			xt = self.lookupTable:forward(it)
		else
			if inferenceMax == 1 then
				inferenceProbs, it = torch.max(logprobs, 2)
				it = it:view(-1):long()
			else
				local prob_prev
				if temperature == 1.0 then
					prob_prev = torch.exp(logprobs)
				else
					prob_prev = torch.exp(torch.div(logprobs, temperature))
				end
				it = torch.multinomial(prob_prev, 1)
				inferenceProbs = logprobs:gather(2, it)
				it = it:view(-1):long()
			end
			xt = self.lookupTable:forward(it)
		end

		if t >= 3 then
			output[t-2] = it
			outputProbs[t-2] = inferenceProbs:view(-1):float()
		end

		local inputs = {xt, unpack(state)}
		local out = self.cell:forward(inputs)
		logprobs = out[self.numState+1]
		state = {}
		for i = 1, self.numState do table.insert(state, out[i]) end
	end

	output = output:transpose(1, 2):contiguous()
	outputProbs = outputProbs:transpose(1, 2)

	return output, outputProbs
end