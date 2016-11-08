local M = {}

local Cider = torch.class('Cider', M)

function Cider:__init(n, sigma)
	self.n = n
	self.sigma = sigma
end

local function precook(seq, n)
	local seqLength = seq:size(1)
	local counts = {}
	for k = 1, n do
		for i = 1, seqLength-k+1 do
			local flag = 0
			for j = 1, k do
				if seq[i+j-1] == 0 then
					flag = 1
					break
				end
			end
			if flag == 0 then
				local ngram = seq[{{i, i+k-1}}]
				if not counts[ngram] then
					counts[ngram] = 1
				else
					counts[ngram] = counts[ngram] + 1
				end
			else
				break
			end
		end
	end
	return counts
end

local function refsCook(refs, n)
	local out = {}
	local numRefs = refs:size(1)
	for i = 1, numRefs do
		table.insert(out, precook(refs[i], n))
	end
	return out
end

local function testCook(test, n)
	return precook(test, n)
end

function Cider:append(refs, test)
	if not self.crefs then self.crefs={} end
	if not self.ctest then self.ctest={} end
	assert(refs ~= nil, '')
	assert(test ~= nil, '')
	table.insert(self.crefs, refsCook(refs, self.n))
	table.insert(self.ctest, testCook(test, self.n))
end

function Cider:computeDocFreq()
	self.documentFrequency = {}
	local maxFre = -1
	for _, refs in pairs(self.crefs) do
		local ngrams = {}
		for _, ref in pairs(refs) do
			for ngram, _ in pairs(ref) do
				if not ngrams[ngram] then
					table.insert(ngrams, ngram)
				end
			end
		end
		for _, ngram in pairs(ngrams) do
			if not self.documentFrequency[ngram] then
				self.documentFrequency[ngram] = 1
			else
				self.documentFrequency[ngram] = self.documentFrequency[ngram] + 1
			end
		end
	end
	for _, v in pairs(self.documentFrequency) do
		if v > maxFre then
			maxFre = v
		end
	end
	assert(#self.crefs >= maxFre, 'DF wrong when calculating CIDEr Score')
end

function Cider:counts2vec(cnts)
	local vec = {}
	local norm = torch.Tensor(self.n):zero()
	for i = 1, self.n do
		local dict = {}
		table.insert(vec, dict)
	end
	local length = 0
	for ngram, termFreq in pairs(cnts) do
		local df = 0.0
		if self.documentFrequency[ngram] then
			df = math.log(self.documentFrequency[ngram])
		end
		local n = ngram:size(1)
		vec[n][ngram] = (self.refLen - df) * termFreq
		norm[n] = norm[n] + (vec[n][ngram])^2
		
		if n == 1 then
			length = length + termFreq
		end
	end
	norm = torch.sqrt(norm)
	return vec, norm, length
end

function Cider:sim(vecTst, vecRef, normTst, normRef, lengthTst, lengthRef)
	local delta = lengthTst - lengthRef
	local val = torch.Tensor(self.n):zero()
	for i = 1, self.n do
		for ngram, _ in pairs(vecTst[i]) do
			if vecRef[i][ngram] then
				val[i] = val[i] + min(vecTst[i][ngram], vecRef[i][ngram]) * vecRef[i][ngram]
			end
		end
		if (normTst[i] ~= 0) and (normRef[i] ~= 0) then
			val[i] = val[i] / (normTst[i] * normRef[i])
		end
		val[i] = val[i] * math.exp(-(delta^2)/(2*(self.sigma)^2))
	end
	return val
end

function Cider:computeCider()
	self.refLen = math.log(#self.crefs)
	local scores = torch.Tensor(#self.crefs):zero()
	for i = 1, #self.crefs do
		local test = self.ctest[i]
		local refs = self.crefs[i]
		local vec, norm, length = self:counts2vec(test)
		local score = torch.FloatTensor(self.n):zero()
		for _, ref in pairs(refs) do
			local vecRef, normRef, lengthRef = self:counts2vec(ref)
			local val = self:sim(vec, vecRef, norm, normRef, length, lengthRef)
			score = torch.add(score, val)
		end
		local scoreAvg = torch.mean(score)
		scoreAvg = scoreAvg / (#refs)
		scoreAvg = scoreAvg * 10.0
		scores[i] = scoreAvg
	end
	return scores
end

function Cider:computeScore()
	self:computeDocFreq()
	local scores = self:computeCider()
	return torch.mean(scores), scores:totable()
end

return M.Cider