--[[
--
--  code from 'https://github.com/karpathy/neuraltalk2/blob/master/misc/LanguageModel.lua'
--
--]]
require 'nn'

local crit, parent = torch.class('nn.SeqCrossEntropyCriterion', 'nn.Criterion')

function crit:__init()
	parent:__init(self)
	self.gradInput = torch.Tensor()
end

--[[
	input is a Tensor of size batchsize * (seqLength+2) * (vocabSize+1)
	seq is a LongTensor of size batchsize * (seqLength+1).
	The way we infer the target in this criterion is as follows:
	- at first time step the output is ignored (loss = 0). It's the image tick
	- the label sequence "seq" is shifted by one to produce targets
	- at last time step the output is always the special END token (last dimension)
	The criterion must be able to accomodate variably-sized sequences by making sure the gradients are properly set to zeros where appropriate.
--]]
function crit:updateOutput(input, seq)
	self.gradInput:resizeAs(input):zero()
	local batchsize, L, Mp1 = input:size(1), input:size(2), input:size(3)

	local seqLength = seq:size(2)-1
	assert(seqLength == L-2, 'input Tensor should be 2 larger in time')

	local loss = 0
	local n = 0
	for b = 1, batchsize do -- iterate over batches
		local first_time = true
		for t = 2, L do -- iterate over sequence time (ignore t=1, dummy forward for the image)

			-- fetch the index of the next token in the sequence
			local target_index
			if t > seqLength then -- we are out of bounds of the index sequence: pad with null tokens
				target_index = 0
			else
				target_index = seq[{b, t}]
			end
			-- the first time we see null token as next index, actually want the model to predict the END token
			if target_index == 0 and first_time then
				target_index = Mp1
				first_time = false
			end

			-- if there is a non-null next token, enforce loss!
			if target_index ~= 0 then
				-- accumulate loss
				loss = loss - input[{ b,t,target_index }] -- log(p)
				self.gradInput[{ b, t, target_index }] = -1
				n = n + 1
			end

		end
	end
	self.output = loss / n -- normalize by number of predictions that were made
	self.gradInput:div(n)
	return self.output
end

function crit:updateGradInput(input, seq)
	return self.gradInput
end
