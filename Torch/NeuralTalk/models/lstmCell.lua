--[[
--
--  code from 'https://github.com/karpathy/neuraltalk2/blob/master/misc/LSTM.lua'
--
--]]
local nn = require 'nn'
require 'nngraph'

local function lstmCell(inputSize, outputSize, hidenStateSize, rnnDepth, dropout)
	dropout = dropout or 0

	-- there will be 2*rnnDepth+1 inputs
	-- input, prev_h * rnnDepth, prev_c * rnnDepth
	local inputs = {}
	for L = 1, 2*rnnDepth+1 do
		table.insert(inputs, nn.Identity()())
	end

	local x, xSize
	local outputs = {}
	for L = 1, rnnDepth do
		-- c, h from previous timesteps
		local prevH = inputs[L*2+1]
		local prevC = inputs[L*2]
		-- set the input to this time step layer
		if L == 1 then
			x = inputs[1]
			xSize = inputSize
		else
			x = outputs[(L-1)*2] -- prev H (when sample) or actual input (when forward - train)
			if dropout > 0 then
				x = nn.Dropout(dropout)(x):annotate{name='dropout_' .. L}
			end
			xSize = hidenStateSize
		end
		-- evaluate the input sums at once for efficiency
		-- inputSum = W*(x, h)
		local x2h = nn.Linear(xSize, 4 * hidenStateSize)(x):annotate{name='x2h_' .. L}
		local h2h = nn.Linear(hidenStateSize, 4*hidenStateSize)(prevH):annotate{name='h2h_' .. L}
		local inputSum = nn.CAddTable()({x2h, h2h})

		local reshapeSum = nn.Reshape(4, hidenStateSize)(inputSum)
		local n1, n2, n3, n4 = nn.SplitTable(1, 2)(reshapeSum):split(4)
		local iGate = nn.Sigmoid()(n1)
		local fGate = nn.Sigmoid()(n2)
		local oGate = nn.Sigmoid()(n3)
		local gGate = nn.Tanh()(n4)

		local nextC = nn.CAddTable()({
			nn.CMulTable()({fGate, prevC}),
			nn.CMulTable()({iGate, gGate})
			})
		local nextH = nn.CMulTable()({
			oGate,
			nn.Tanh()(nextC)
			})

		table.insert(outputs, nextC)
		table.insert(outputs, nextH)
	end

	-- set up the decoder
	local topH = outputs[#outputs]
	if dropout > 0 then
		topH = nn.Dropout(dropout)(topH):annotate{name='dropout_top'}
	end
	local proj = nn.Linear(hidenStateSize, outputSize)(topH):annotate{name='decoder'}
	local logsoft = nn.LogSoftMax()(proj)
	table.insert(outputs, logsoft)

	return nn.gModule(inputs, outputs)
end

return lstmCell