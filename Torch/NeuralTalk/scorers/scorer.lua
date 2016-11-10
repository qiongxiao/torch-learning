local M = {}

local Scorer = torch.class('Scorer', M)

function Scorer:__init(scorerType)
	if scorerType == 'CIDEr' then
		local Cider = require 'scorers.cider'
		self.scorer = Cider(4, 6.0)
		self.type = scorerType
	elseif scorerType == 'BLEU' then
		local BLEU = require 'scorers.bleu'
		self.scorer = BLEU(4)
		self.type = scorerType
	else
		error('invalid scorer type')
	end
end

function Scorer:update(refs, test)
	self.scorer:append(refs, test)
end

function Scorer:computeScore(option)
	if self.scorerType ~= 'BLEU' then
		return self.scorer:computeScore()
	else
		return self.scorer:computeScore(option)
	end
end

return M.Scorer