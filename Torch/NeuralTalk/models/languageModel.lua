--[[
--
--  code from 'https://github.com/karpathy/neuraltalk2/blob/master/misc/LanguageModel.lua'
--
--]]

require 'nn'
require 'models.lstm'
require 'models.recursiveLookupTable'

local layer, parent = torch.class('nn.LanguageModel', 'nn.Module')

function layer:__init(opt, vocabSize)
	parent.__init(self)

	self.vocabSize = vocabSize
	self.inputEncodingSize = opt.inputEncodingSize
	self.rDepth = opt.rDepth
	local dropout = opt.dropout
	self.seqLength = opt.seqLength
	self.hiddenStateSize = opt.hiddenStateSize
	local skipFlag = opt.skipFlag
	self.lstm = nn.LSTM(self.inputEncodingSize, self.vocabSize+1, self.hiddenStateSize, self.seqLength+2, self.rDepth, dropout, skipFlag)
	self.rLookupTable = nn.RLookupTable(self.vocabSize+1, self.inputEncodingSize, self.seqLength+1, skipFlag)
end

function layer:getModulesList()
	return {self.lstm:getModule(), self.rLookupTable:getModule()}
end

function layer:parameters()
	local p1,g1 = self.lstm:parameters()
	local p2,g2 = self.rLookupTable:parameters()

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
	self.rLookupTable:training()
end

function layer:evaluate()
	self.lstm:evaluate()
	self.rLookupTable:evaluate()
end

function layer:updateOutput(input)
	local imgs = input[1]
	local seq = input[2]

	assert(imgs:size(2) == self.inputEncodingSize)
	assert(seq:size(1) == self.seqLength)

	local batch_size = seq:size(2)
	self.rLookupTableInput = tortch.cat({torch.Tensor(1, batch_size):fill(self.vocabSize+1), seq}, 1)
	local seqEncoded = self.rLookupTable:forward(seqWithStart)
	self.lstmInput = torch.cat({imgs:view(1, batch_size, self.inputEncodingSize), seqEncoded}, 1)
	self.output = self.lstm:forward(lstmInput)
	return self.output
end

function layer:updateGradInput(input, gradOutput)
	local dImgs
	local dlstmInput = self.lstm:backward(self.lstmInput, gradOutput)
	local drLookupTableInput = self.rLookupTable:backward(self.rLookupTableInput, dlstmInput:narrow(1, 2, self.seqLength+1))
	dImgs = dlstmInput[1]:view(input[1]:size())
	self.gradInput = {dImgs, torch.Tensor()}
	return self.gradInput
end