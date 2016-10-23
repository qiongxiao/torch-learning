require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'optim'

require 'CNNModel'
require 'DataLoader'

local utils = require 'utils'
local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-batch_size', 50)
cmd:option('-preprocess', 'simple')
cmd:option('-train_full', 1)
-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
cmd:option('-model_type', 'VGG')
cmd:option('-dropout', 0.5)
cmd:option('-conv_dropout', 0.4)
cmd:option('-batchnorm', 1)
cmd:option('-spatial_batchnorm', 1)
cmd:option('-data_flip', 0)
-- Optimization options
cmd:option('-max_epochs', 50)
cmd:option('-learning_rate', 5e-3)
cmd:option('-lr_decay_every', 5)
cmd:option('-lr_decay_factor', 0.5)
cmd:option('-max_decrease_iters', 0)
-- Output options
cmd:option('-print_every', 1)
cmd:option('-checkpoint_every', 900)
cmd:option('-checkpoint_name', 'cv/checkpoint')

local opt = cmd:parse(arg)

local dtype = 'torch.CudaTensor'
local classes = {'airplane', 'automobile', 'bird', 'cat', 
				 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

-- Initialize the model and criterion
local model = nil
local start_i = 0

if opt.init_from ~= '' then
	print('Initializing from ', opt.init_from)
	local checkpoint = torch.load(opt.init_from)
	model = checkpoint.model:type(dtype)
	if opt.reset_iterations == 0 then
		start_i = checkpoint.i
	end
else
	model = nn.CNNModel(opt, #classes):type(dtype)
end

local params, grad_params = model:getParameters()
-- Loss functions: Softmax
local crit = nn.CrossEntropyCriterion():type(dtype)

local loader = DataLoader(opt)

local function f( w )
	assert(w == params)
	grad_params:zero()

	-- Get a minibatch and run the model forward
	local x, y = loader:nextBatch('train')
	x, y = x:type(dtype), y:type(dtype)
	local scores = model:forward(x)
	local loss = crit:forward(scores, y)

	local grad_scores = crit:backward(scores, y)
	model:backward(x, grad_scores)

	return loss, grad_params
end

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
	-- reset model state as 'train'
	model:training()
	return eval_loss, eval_accuracy
end

local optim_config = {
	learningRate = opt.learning_rate
}
local num_train = loader.split_sizes['train']
local num_iterations = opt.max_epochs * num_train
local train_loss_history = {}
local val_loss_history = {}
local val_loss_history_it = {}
local last_val_accuracy = 0
local decreasing = 0
local plotter = Plotter('plot/out.json')
paths.mkdir(paths.dirname('plot/out.json'))
plotter:info({created_time=io.popen('date'):read(), tag='Plot'})

-- set model state as 'train'
model:training()
for i = start_i + 1, num_iterations do
	local epoch = math.floor(i / num_train) + 1
  -- Check if we are at the end of an epoch
	if i % num_train == 0 then
		-- decay learning rate
		if epoch % opt.lr_decay_every == 0 then
			local old_lr = optim_config.learningRate
			optim_config = {learningRate = old_lr * opt.lr_decay_factor}
		end
		-- avoid overfit
		if opt.max_decrease_iters > 0 then
			local _, val_accuracy = eval('val')
			if val_accuracy < last_val_accuracy then
				if decreasing > opt.max_decrease_iters then break end
				decreasing = decreasing + 1
			else
				decreasing = 0
			end
			last_val_accuracy = val_accuracy
		end
	end
	-- Take a gradient step and maybe print
	-- Note that adam returns a singleton array of losses
	local _, loss = optim.adam(f, params, optim_config)
	table.insert(train_loss_history, loss[1])
	plotter:add('Loss', 'Train', #train_loss_history, train_loss_history[#train_loss_history])
	if opt.print_every > 0 and i % opt.print_every == 0 then
		local float_epoch = i / num_train + 1
		local msg = 'Epoch %.2f / %d, i = %d / %d, loss = %f'
		local args = {msg, float_epoch, opt.max_epochs, i, num_iterations, loss[1]}
		print(string.format(unpack(args)))
	end
	-- Maybe save a checkpoint
	local check_every = opt.checkpoint_every
	if (check_every > 0 and i % check_every == 0) or i == num_iterations then
		-- Evaluate loss on the validation set.
		local val_loss, val_accuracy = eval('val')
		table.insert(val_loss_history, val_loss)
		table.insert(val_loss_history_it, i)
		-- reset model state as 'train'
		model:training()
		-- First save a JSON checkpoint, excluding the model
		local checkpoint = {
			opt = opt,
			train_loss_history = train_loss_history,
		 	val_loss_history = val_loss_history,
			val_loss_history_it = val_loss_history_it,
			i = i
		}
		local filename = string.format('%s.json', opt.checkpoint_name)
		-- Make sure the output directory exists before we try to write it
		paths.mkdir(paths.dirname(filename))
		utils.write_json(filename, checkpoint)
		plotter:add('Loss', 'Validation', #train_loss_history, val_loss_history[#val_loss_history])
		plotter:add('Accuracy', 'Validation', #val_loss_history, val_accuracy)
		model:float()
		checkpoint.model = model
		local filename = string.format('%s_%d.t7', opt.checkpoint_name, i)
		paths.mkdir(paths.dirname(filename))
		torch.save(filename, checkpoint)
		model:type(dtype)
		params, grad_params = model:getParameters()
	end
	collectgarbage()
end
