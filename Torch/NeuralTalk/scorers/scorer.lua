local M = {}

local Scorer = torch.class('Scorer', M)

function Scorer:__init(scorerType)
	if scorerType == 'CIDEr' then
		local Cider = require 'scorers.cider'
		self.scorer = Cider(4, 6.0)
		self.type = scorerType
	else
		error('invalid scorer type')
	end
end

function Scorer:update(refs, test)
	if self.type == 'CIDEr' then
		self.scorer:append(refs, test)
	else
		error('invalid scorer when updating scorer')
	end
end

function Scorer:computeScore()
	return self.scorer:computeScore()
end

return M.Scorer