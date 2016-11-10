local M = {}

local BLEU = torch.class('BLEU', M)

function BLEU:__init(n)
	self.n = n
end

local function precook(seq, n)
	local seqLength = seq:size(1)
	local realSeqLength = 0
	for i = 1, seqLength do
		if seq[i] == 0 then
			break
		end
		realSeqLength = realSeqLength + 1
	end

	local counts = {}
	for k = 1, n do
		for i = 1, realSeqLength-k+1 do
			local flag = 0
			local ngram = seq[i]
			for j = 1, k-1 do
				ngram = ngram .. ' ' .. seq[i+j]
			end
			if not counts[ngram] then
				counts[ngram] = {}
				counts[ngram][1] = 1
				counts[ngram][2] = k
			else
				counts[ngram][1] = counts[ngram][1] + 1
			end
		end
	end
	return realSeqLength, counts
end

local function refsCook(refs, n)
	local maxcounts = {}
	local numRefs = refs:size(1)
	local reflen = torch.Tensor(numRefs)
	for i = 1, numRefs do
		local rl, counts = precook(refs[i], n)
		reflen[i] = rl
		for ngram, v in pairs(counts) do
			if not maxcounts[ngram] then
				maxcounts[ngram] = v
			else
				maxcounts[ngram][1] = math.max(maxcounts[ngram][1], v[1])
			end
		end
	end
	return {reflen, maxcounts}
end

local function testCook(test, n, cookedRefs)
	local reflen, refmaxcounts = cookedRefs[1], cookedRefs[2]
	local testlen, counts = precook(test, n)
	local result = {}

	result['reflen'] = reflen
	result['testlen'] = testlen
	result['guess'] = {}
	result['correct'] = {}
	for i = 1, n do
		result['guess'][i] = math.max(0, testlen-i+1)
		table.insert(result['correct'], 0)
	end

	for ngram, v in pairs(counts) do
		local count = v[1]
		local refcount = refmaxcounts[ngram] and refmaxcounts[ngram][1] or 0
		result['correct'][v[2]] = result['correct'][v[2]] + math.min(refcount, count)
	end

	return result
end

function BLEU:append(refs, test)
	if not self.crefs then self.crefs={} end
	if not self.ctest then self.ctest={} end
	assert(refs ~= nil, '')
	assert(test ~= nil, '')
	table.insert(self.crefs, refsCook(refs, self.n))
	table.insert(self.ctest, testCook(test, self.n, self.crefs[#self.crefs]))
end

local function _singleReflen(reflens, option, testlen)
	local reflen
	if option == 'shortest' then
		reflen = torch.min(reflens)
	elseif option == 'average' then
		reflen = torch.mean(reflens)
	elseif option == 'closest' then
		local reflenSize = reflens:size(1)
		local minDif = math.abs(reflens[1]-testlen)
		reflen = reflens[1]
		for i = 2, reflenSize do
			local dif = math.abs(reflens[i]-testlen)
			if minDif > dif then
				minDif = dif
				reflen = reflens[i]
			end
		end
	else
		error('invalid reflen option')
	end
	return reflen
end


function BLEU:computeScore(option)
	local n = self.n
	local small = 1e-9
	local tiny = 1e-15
	local bleuList = {}
	for i = 1, n do
		table.insert(bleuList, {})
	end

	if not option then
		option = (#self.crefs == 1) and 'average' or 'closest'
	end

	self._testlen = 0
	self._reflen = 0

	local totalcomps = {testlen=0, reflen=0, guess={}, correct={}}
	for i = 1, n do
		table.insert(totalcomps['guess'], 0)
		table.insert(totalcomps['correct'], 0)
	end

	for _, comps in pairs(self.ctest) do
		local testlen = comps['testlen']
		self._testlen = self._testlen + testlen

		local reflen = _singleReflen(comps['reflen'], option, testlen)

		self._reflen = self._reflen + reflen

		for _, v in ipairs({'guess', 'correct'}) do
			for k = 1, n do
				totalcomps[v][k] = totalcomps[v][k] + comps[v][k]
			end
		end

		local bleu = 1
		for k = 1, n do
			bleu = bleu * (comps['correct'][k] + tiny)/(comps['guess'][k] + small)
			table.insert(bleuList[k], bleu^(1/(k+1)))
		end
		local ratio = (testlen + tiny) / (reflen + small)
		if ratio < 1 then
			for k = 1, n do
				bleuList[k][#bleuList[k]] = bleuList[k][#bleuList[k]] * math.exp(1 - 1/ratio)
			end
		end
	end

	totalcomps['reflen'] = self._reflen
	totalcomps['testlen'] = self._testlen

	local bleus = {}
	bleu = 1
	for k = 1, n do
		bleu = bleu * (totalcomps['correct'][k] + tiny)/(totalcomps['guess'][k] + small)
		table.insert(bleus, bleu^(1/(k+1)))
	end
	local ratio = (self._testlen + tiny) / (self._reflen + small)
	if ratio < 1 then
		for k = 1, n do
			bleus[k] = bleus[k] * math.exp(1 - 1/ratio)
		end
	end

	return bleus, bleuList

end

return M.BLEU