--[[
--
--  code from 'https://github.com/karpathy/neuraltalk2/blob/master/misc/LanguageModel.lua'
--
--]]

require 'cutorch'
local nn = require 'nn'
local cudnn = require 'cudnn'

local lstmCell = require 'models.lstmCell'
require 'models.expander'
require 'models.dynamicLSTM'



local layer, parent = torch.class('nn.FeatureToSeq', 'nn.Module')

function layer:__init(opt, nFeatures, vocabSize)
	parent.__init(self)

	self.nFeatures = nFeatures
	self.linear = nn.Sequential():add(Linear(nFeatures, opt.encodingSize)):add(cndnn:ReLU(true))
	self.linear:get(1).bias:zero()
	self.expander = nn.Expander(opt.seqPerImg)

	self.vocabSize = vocabSize
	self.encodingSize = opt.encodingSize
	self.rDepth = opt.rDepth
	self.seqLength = opt.seqLength
	self.hiddenStateSize = opt.hiddenStateSize

	self.lstmCell = lstmCell(self.encodingSize, self.vocabSize+1, self.hiddenStateSize, self.rDepth, opt.lstmDropout)
	self.lstm = nn.dynamicLSTM(self.lstmCell, self.encodingSize, self.vocabSize+1, self.hiddenStateSize, self.seqLength+2, self.rDepth)

	self.lookupTable = nn.LookupTable(self.vocabSize+1, self.encodingSize)

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
				self.initState[h]:resize(batchsize, self.hiddenStateSize):zero() -- expand the memory
			end
		else
			self.initState[h] = torch.zeros(batchsize, self.hiddenStateSize):cuda()
		end
	end
	self.numState = #self.initState
end

function layer:getModulesList()
	return {self.linear, self.expander, self.lstmCell, self.lookupTable}
end

function layer:parameters()
	local p0,g0 = self.linear:parameters()
	local p1,g1 = self.lstmCell:parameters()
	local p2,g2 = self.lookupTable:parameters()

	local params = {}
	for k,v in pairs(p0) do table.insert(params, v) end
	for k,v in pairs(p1) do table.insert(params, v) end
	for k,v in pairs(p2) do table.insert(params, v) end

	local grad_params = {}
	for k,v in pairs(g0) do table.insert(grad_params, v) end
	for k,v in pairs(g1) do table.insert(grad_params, v) end
	for k,v in pairs(g2) do table.insert(grad_params, v) end

	-- ################ ?? destroy weight sharing?? ################
	return params, grad_params
end

function layer:training()
	self.linear:training()
	self.expander:training()
	self.lstm:training()
	self.lookupTable:training()
end

function layer:evaluate()
	self.linear:evaluate()
	self.expander:evaluate()
	self.lstm:evaluate()
	self.lookupTable:evaluate()
end

--[[
	input is a table of:
		1.	torch.Tensor of size "batchsize(unexpanded) * nFeatures"
		2.	torch.LongTensor of size "batchsize(expanded) * (seqLength+1)", elements 1..M
			where M = vocabSize
	output:
		torch.Tensor of size "batchsize * (seqLength+2) * (M+1)"
--]]
function layer:updateOutput(input)
	local imgs = input[1]
	local seq = input[2]
	assert(imgs:size(2) == self.nFeatures)
	assert(seq:size(2) == self.seqLength+1)

	-- size "batchsize(unexpanded) * nFeatures" --> size "batchsize(expanded) * encodingSize"
	self.linear:forward(imgs)
	local feats = self.expander:forward(self.linear.output)

	local batchsize = seq:size(1)

	--torch.cat({torch.LongTensor(batchsize, 1):fill(self.vocabSize+1), seq})
	local seqEncoded = self.lookupTable:forward(seq)
	-- seqEncoded's size is "batchsize * (seqLength+1) * encodingSize"
	self.lstmInput = torch.cat({feats:view(batchsize, 1, self.encodingSize), seqEncoded}, 2)
	-- lstmInput's size is "batchsize * (seqLength+2) * encodingSize"
	self.lstmInput = self.lstmInput:transpose(1, 2)
	-- lstmInput's size is "(seqLength+2) * batchsize * encodingSize"
	self.output = self.lstm:forward(self.lstmInput)
	self.output = self.output:transpose(1, 2)

	return self.output
end

function layer:updateGradInput(input, gradOutput)
	local dFeats
	local dlstmInput = self.lstm:backward(self.lstmInput, gradOutput:transpose(1, 2))
	local dlookupTableInput = self.lookupTable:backward(input[2], dlstmInput:narrow(1, 2, self.seqLength+1):transpose(1, 2))
	self.expander:backward(self.linear.output, dlstmInput[1])
	dImgs = self.linear:backward(input[1], self.expander.gradInput)
	self.gradInput = {dImgs, torch.Tensor()}
	return self.gradInput
end

--[[
	input:
		torch.Tensor of size "batchsize * nFeatures"
	output:
		torch.Tensor of size "batchsize * seqLength"
--]]
function layer:inference(imgs)
	local inferenceMax = self.inferenceMax
	local temperature = self.temperature

	-- size "batchsize * nFeatures" --> size "batchsize * encodingSize"
	local feats = self.linear:forward(imgs)

	local batchsize = feats:size(1)
	self:_createInitState(batchsize)
	local state = self.initState

	local output = torch.LongTensor(self.seqLength, batchsize):zero()
	local outputProbs = torch.FloatTensor(self.seqLength, batchsize)
	local logprobs

	for t = 1, self.seqLength+2 do

		local xt, it, inferenceProbs

		if t == 1 then
			xt = feats
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

		local out = self.lstmCell:forward(inputs)
		logprobs = out[self.numState+1]
		state = {}
		for i = 1, self.numState do table.insert(state, out[i]) end
	end

	output = output:transpose(1, 2):contiguous()
	outputProbs = outputProbs:transpose(1, 2)

	return output, outputProbs
end
