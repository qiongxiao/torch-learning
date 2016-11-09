--[[
--
--  feature to sequence model (cannot skip some null tokens when training)
--
--]]
require 'nn'

require 'models.expander'
require 'models.dynamicLSTM'

local layer, parent = torch.class('nn.FeatureToSeq', 'nn.Module')

function layer:__init(opt, vocabSize)
	parent.__init(self)

	-- For thin model
	self.opt = opt
	self.vocabSize = vocabSize

	self.expander = nn.Expander(opt.seqPerImg)

	self.encodingSize = opt.encodingSize
	self.rDepth = opt.rDepth
	self.seqLength = opt.seqLength
	self.hiddenStateSize = opt.hiddenStateSize

	self.lstm = nn.dynamicLSTM(self.encodingSize, self.vocabSize+1, self.hiddenStateSize, self.seqLength+2, self.rDepth, opt)

	self.lookupTable = nn.LookupTable(self.vocabSize+1, self.encodingSize)

	self.inferenceMax = opt.inferenceMax
	self.temperature = opt.temperature

end

function layer:getModulesList()
	return {self.expander, self.lstm.cell, self.lookupTable}
end

function layer:createSlices()
	self.lstm:createSlices()
end

function layer:shareSlices()
	self.lstm:shareSlices()
end

function layer:parameters()
	local p1,g1 = self.lstm:parameters()
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
	self.expander:training()
	self.lstm:training()
	self.lookupTable:training()
end

function layer:evaluate()
	self.expander:evaluate()
	self.lstm:evaluate()
	self.lookupTable:evaluate()
end

--[[
	input is a table of:
		1.	torch.Tensor of size "batchsize(unexpanded) * encodingSize"
		2.	torch.LongTensor of size "batchsize(expanded) * (seqLength+1)", elements 1..M
			where M = vocabSize
	output:
		torch.Tensor of size "batchsize * (seqLength+2) * (M+1)"
--]]
function layer:updateOutput(input)
	local imgs = input[1]
	local seq = input[2]
	assert(imgs:size(2) == self.encodingSize)
	assert(seq:size(2) == self.seqLength+1)

	-- size "batchsize(unexpanded) * encodingSize" --> size "batchsize(expanded) * encodingSize"
	local feats = self.expander:forward(imgs)

	local batchsize = seq:size(1)

	-- seqEncoded's size is "batchsize * (seqLength+1) * encodingSize"
	seq[torch.eq(seq, 0)] = self.vocabSize+1
	local seqEncoded = self.lookupTable:forward(seq)
	-- lstmInput's size is "batchsize * (seqLength+2) * encodingSize"
	self.lstmInput = torch.cat({feats:view(batchsize, 1, self.encodingSize), seqEncoded}, 2)
	-- lstmInput's size is "(seqLength+2) * batchsize * encodingSize"
	self.lstmInput = self.lstmInput:transpose(1, 2)

	self.output = self.lstm:forward(self.lstmInput)
	self.output = self.output:transpose(1, 2)

	return self.output
end

function layer:updateGradInput(input, gradOutput)
	local dlstmInput = self.lstm:backward(self.lstmInput, gradOutput:transpose(1, 2))
	local dlookupTableInput = self.lookupTable:backward(input[2], dlstmInput:narrow(1, 2, self.seqLength+1):transpose(1, 2))
	self.expander:backward(input[1], dlstmInput[1])
	self.gradInput = {self.expander.gradInput, torch.Tensor()}
	return self.gradInput
end

--[[
	input:
		torch.Tensor of size "batchsize * encodingSize"
	output:
		torch.Tensor of size "batchsize * seqLength"
--]]
function layer:inference(imgs)

	local output, outputProbs = self.lstm:inference(imgs, self.lookupTable)

	output = output:narrow(1, 2, self.seqLength):transpose(1, 2)
	outputProbs = outputProbs:narrow(1, 2, self.seqLength):transpose(1, 2)
	output[torch.eq(output, self.vocabSize+1)] = 0
	
	return output, outputProbs
end
