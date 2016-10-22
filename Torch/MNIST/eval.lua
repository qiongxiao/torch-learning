require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'

require 'LeNetModel'
require 'DataLoader'

function eval( dataset )
	-- set model state as 'test'
	model:evaluate()
	local num_eval = loader.split_sizes[dataset]
	local eval_loss = 0
	for j = 1, num_eval do
		local xv, yv = loader:nextBatch(dataset)
		xv, yv = xv:type(dtype), yv:type(dtype)
		local scores = model:forward(xv)
		local _, indices = torch.max(scores, 2)
		indices:add(-1)
		count = count + indices:eq(yv):sum()
		eval_loss = eval_loss + crit:forward(scores, yv)
	end
	eval_loss = eval_loss / num_eval
	local eval_accuracy = count / loader.set_sizes[dataset]
	print(string.format('%s_loss = %f', dataset, eval_loss))
	print(string.format('%s_accuracy = %f', dataset, eval_accuracy))
	-- reset model state as 'train'
	model:training()
	return eval_loss, eval_accuracy
end