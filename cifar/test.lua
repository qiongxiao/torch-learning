require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'optim'

require 'CNNModel'
require 'DataLoader'

local utils = require 'utils'
local cmd = torch.CmdLine()

-- Model options
cmd:option('-checkpoint', 'cv/checkpoint_20000.t7')
local opt = cmd:parse(arg)

local dtype = 'torch.CudaTensor'
local checkpoint = torch.load(opt.checkpoint)

local model = checkpoint.model:type(dtype)
local crit = nn.CrossEntropyCriterion():type(dtype)
local loader = DataLoader(checkpoint.opt)

local function eval( dataset )
	-- set model state as 'test'
	model:evaluate()
	local num_eval = loader.split_sizes[dataset]
	local eval_loss = 0
	local count = 0
	for j = 1, num_eval do
		local xv, yv = loader:nextBatch(dataset)
		xv, yv = xv:type(dtype), yv:type(dtype)
		local scores = model:forward(xv)
		local _, indices = torch.max(scores, 2)
		eval_loss = eval_loss + crit:forward(scores, yv)
		count = count + indices:eq(yv:type('torch.CudaLongTensor')):sum()
	end
	eval_loss = eval_loss / num_eval
	local eval_accuracy = count / loader.set_sizes[dataset]
	print(string.format('%s_loss = %f', dataset, eval_loss))
	print(string.format('%s_accuracy = %f', dataset, eval_accuracy))
	return eval_loss, eval_accuracy
end

local _, test_accuracy = eval('test')
