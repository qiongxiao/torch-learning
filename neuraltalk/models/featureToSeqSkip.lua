--[[
--
--  code adaption from 'https://github.com/karpathy/neuraltalk2/blob/master/misc/LanguageModel.lua'
--
--]]
require 'nn'

local lstmCell = require 'models.lstmCell'
require 'models.expander'

local layer, parent = torch.class('nn.FeatureToSeqSkip', 'nn.Module')

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
	self.lstmCell = lstmCell(self.encodingSize, self.vocabSize+1, self.hiddenStateSize, self.rDepth, opt.lstmDropout)

	self.lookupTable = nn.LookupTable(self.vocabSize+1, self.encodingSize)

	self.inferenceMax = opt.inferenceMax
	self.temperature = opt.temperature

	self:_createInitState(1)

end

function layer:_createInitState(batchsize)
	assert(batchsize ~= nil, 'batch size for FeatureToSeqSkip model must be provided')
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

function layer:createSlices()
	self.slices = { self.lstmCell }
	self.lookupTables = { self.lookupTable }
	for t = 2, self.seqLength+2 do
		self.slices[t] = self.lstmCell:clone('weight', 'bias', 'gradWeight', 'gradBias')
		self.lookupTables[t] = self.lookupTable:clone('weight', 'gradWeight')
	end
end

function layer:shareSlices()
	assert(self.slices ~= nil, 'cannot share before clone')
	assert(self.lookupTables ~= nil, 'cannot share before clone')
	for t = 1, self.seqLength+2 do
		self.slices[t]:share(self.lstmCell, 'weight', 'bias', 'gradWeight', 'gradBias')
		self.lookupTables[t]:share(self.lookupTable, 'weight', 'gradWeight')
	end
end

function layer:getModulesList()
	return {self.expander, self.lstmCell, self.lookupTable}
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
	self.expander:training()
	if self.slices == nil then self:createSlices() end -- create these lazily if needed
	--self:shareSlices()
	for k,v in pairs(self.slices) do v:training() end
	for k,v in pairs(self.lookupTables) do v:training() end
end

function layer:evaluate()
	self.expander:evaluate()
	if self.slices == nil then self:createSlices() end -- create these lazily if needed
	--self:shareSlices()
	for k,v in pairs(self.slices) do v:evaluate() end
	for k,v in pairs(self.lookupTables) do v:evaluate() end
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
	-- seq size "(seqLength+1) * batchsize(expanded)"
	local seq = input[2]:transpose(1, 2)
	if self.slices == nil then self:createSlices() end -- create these lazily if needed
	--self:shareSlices()
	assert(imgs:size(2) == self.encodingSize)
	assert(seq:size(1) == self.seqLength+1)

	-- size "batchsize(unexpanded) * encodingSize" --> size "batchsize(expanded) * encodingSize"
	local feats = self.expander:forward(imgs)

	local batchsize = feats:size(1)
	self.output:resize(self.seqLength+2, batchsize, self.vocabSize+1)

	self:_createInitState(batchsize)

	self.state = {[0] = self.initState}
	self.inputs = {}
	self.lookupTablesInputs = {}
	self.tmax = 0

	for t = 1, self.seqLength+2 do
		local canSkip = false
		local xt
		if t == 1 then
			xt = feats
		else
			local it = seq[t-1]:clone()
			if torch.sum(it) == 0 then
				canSkip = true
			end
			it[torch.eq(it, 0)] = self.vocabSize+1

			if not canSkip then
				self.lookupTablesInputs[t] = it
				xt = self.lookupTables[t]:forward(it)
			end
		end

		if not canSkip then
			self.inputs[t] = {xt, unpack(self.state[t-1])}
			local out = self.slices[t]:forward(self.inputs[t])
			self.output[t] = out[self.numState+1]
			self.state[t] = {}
			for i=1, self.numState do table.insert(self.state[t], out[i]) end
			self.tmax = t
		end
	end

	self.output = self.output:transpose(1, 2)

	return self.output
end

function layer:updateGradInput(input, gradOutput)
	local dFeats

	local dstate = {[self.tmax] = self.initState}
	for t = self.tmax, 1, -1 do
		local dout = {}
		for k=1, #dstate[t] do table.insert(dout, dstate[t][k]) end
		table.insert(dout, gradOutput[{{},{t}}]:squeeze())
		local dinputs = self.slices[t]:backward(self.inputs[t], dout)
		local dxt = dinputs[1]
		dstate[t-1]={}
		for k=2, self.numState+1 do table.insert(dstate[t-1], dinputs[k]) end

		if t == 1 then
			dFeats = dxt
		else
			local it = self.lookupTablesInputs[t]
			self.lookupTables[t]:backward(it, dxt)
		end
	end
	self.expander:backward(input[1], dFeats)
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
	local inferenceMax = self.inferenceMax
	local temperature = self.temperature

	local feats = imgs
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
	output[torch.eq(output, self.vocabSize+1)] = 0
	
	return output, outputProbs
end
