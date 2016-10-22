require 'torch'
require 'cutorch'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(batch_size)
	local mnist = require 'mnist'

	local trainset = mnist.traindataset()
	local testset = mnist.testdataset()

	local dataset = {
		train = {x=trainset.data[{{1, 50000}}]:double(), y=trainset.label[{{1, 50000}}]}
		val = {x=trainset.data[{{50001, 60000}}]:double(), y=trainset.label[{{50001, 60000}}]}
		test = {x=testset.data, y=testset.label}
	}

	self.batch_size = batch_size
	self.set_sizes = {train=50000, val=10000, test=10000}
	self.x_splits = {}
	self.y_splits = {}
	self.split_sizes = {}

	for split, v in pairs(dataset) do
		local num_examples = v:size(1)
		local num_batches = math.floor(num_examples / batch_size)
		local extra = num_examples - num_batches * batch_size

		for idx = 1, num_batches do
			self.x_splits[split][idx] = v[x][{{(idx-1)*batch_size+1, idx*batch_size}}]:view(batch_size, 1, 28, -1)
			self.y_splits[split][idx] = v[y][{{(idx-1)*batch_size+1, idx*batch_size}}]
		end

		if extra ~= 0 then
			num_batches = num_batches + 1
			self.x_splits[split][num_batches] = v[x][{{num_examples-extra+1,num_examples}}]:view(batch_size, 1, 28, -1)
			self.y_splits[split][num_batches] = v[y][{{num_examples-extra+1,num_examples}}]
		end

		self.split_sizes[split] = num_batches
	end

	self.split_idxs = {train=1, val=1, test=1}

end

function DataLoader:nextBatch( split )
	local idx = self.split_idxs[split]
	assert(idx, 'invalid split ' .. split)
	local x = self.x_splits[split][idx]
	local y = self.y_splits[split][idx]
	if idx == self.split_sizes[split] then
		self.split_idxs[split] = 1
	else
		self.split_idxs[split] = idx + 1
	end
	return x, y
end	