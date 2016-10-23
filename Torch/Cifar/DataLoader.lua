require 'torch'
require 'nn'
require 'image'

local utils = require 'utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(kwargs)
	local batch_size = utils.get_kwarg(kwargs, 'batch_size')
	self.batch_size = batche_size
	local preprocess = utils.get_kwarg(kwargs, 'preprocess')
	local train_full = utils.get_kwarg(kwargs, 'train_full')

	-- first download dataset
	if not paths.dirp('data/cifar-10-batches-t7') then
		if not paths.filep('data/cifar-10-torch.tar.gz') then
			local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
			os.execute('wget -P ./data/ ' .. www)
		end
		os.execute('tar xvf ./data/cifar-10-torch.tar.gz -C ./data')
	end

	print '<data init> data loading'

	local dataset = { train = {}, val={}, test={} }
	local train_size
	local val_size
	local test_size

	if train_full == 1 then
		train_size = 45000
		val_size = 5000
		test_size = 10000
		dataset['train']['x'] = torch.Tensor(train_size, 3072)
		dataset['train']['y'] = torch.Tensor(train_size)
		for i = 1, 4 do
			local set = torch.load('data/cifar-10-batches-t7/data_batch_' .. i .. '.t7', 'ascii')
			dataset['train']['x'][{{(i-1)*10000+1, i*10000}}] = set.data:t()
			dataset['train']['y'][{{(i-1)*10000+1, i*10000}}] = set.labels
		end
		dataset['train']['y'] = dataset['train']['y'] + 1
	else
		train_size = 2000
		val_size = 1000
		test_size = 1000
		local set = torch.load('data/cifar-10-batches-t7/data_batch_1.t7', 'ascii')
		dataset['train']['x'] = set.data:t():double()[{{1, train_size}}]
		dataset['train']['y'] = set.labels[1]:double()[{{1, train_size}}] + 1
	end
	dataset['train']['x'] = dataset['train']['x']:reshape(train_size, 3, 32, 32)

	local set = torch.load('data/cifar-10-batches-t7/data_batch_5.t7', 'ascii')
	dataset['val']['x'] = set.data:t():double()[{{1, val_size}}]:reshape(val_size, 3, 32, 32)
	dataset['val']['y'] = set.labels[1]:double()[{{1, val_size}}] + 1

	set = torch.load('data/cifar-10-batches-t7/test_batch.t7', 'ascii')
	dataset['test']['x'] = set.data:t():double()[{{1, test_size}}]:reshape(test_size, 3, 32, 32)
	dataset['test']['y'] = set.labels[1]:double()[{{1, test_size}}] + 1

	self.set_sizes = { train=train_size, val=val_size, test=test_size }

	collectgarbage()
	-- preprocess the image
	if preprocess == "nnc" then
		-- code from 'https://github.com/torch/demos/blob/master/train-on-cifar/train-on-cifar.lua'
		print '<data init> preprocessing data (color space + normalization)'
		normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
		for split, v in pairs(self.set_sizes) do
			for i = 1, v do
				local rgb = dataset[split]['x'][i]
				local yuv = image.rgb2yuv(rgb)
				yuv[1] = normalization(yuv[{{1}}])
				dataset[split]['x'][i] = yuv
			end
		end
		local mean_u = dataset['train']['x']:select(2,2):mean()
		local std_u = dataset['train']['x']:select(2,2):std()
		local mean_v = dataset['train']['x']:select(2,3):mean()
		local std_v = dataset['train']['x']:select(2,3):std()
		for _, v in pairs(dataset) do
			v['x']:select(2,2):add(-mean_u)
			v['x']:select(2,2):div(std_u)
			v['x']:select(2,3):add(-mean_v)
			v['x']:select(2,3):div(std_v)
		end
	else
		print '<data init> preprocessing data (simple normalization)'
		for i = 1, 3 do
			local mean = dataset['train']['x']:select(2,i):mean()
			local std = dataset['train']['x']:select(2,i):std()
			for _, v in pairs(dataset) do
				v['x']:select(2,i):add(-mean)
				v['x']:select(2,i):div(std)
			end
		end
	end
	collectgarbage()

	-- make batches
	print '<data init> making batches'
	self.x_splits = { train={}, val={}, test={} }
	self.y_splits = { train={}, val={}, test={} }
	self.split_sizes = {}
	self.split_idxs = { train=1, val=1, test=1 }

	for split, v in pairs(dataset) do
		local num_examples = v['x']:size(1)
		local num_batches = math.floor(num_examples / batch_size)
		local extra = num_examples - num_batches * batch_size

		for idx = 1, num_batches do
			table.insert(self.x_splits[split], v['x'][{{(idx-1)*batch_size+1, idx*batch_size}}])
			table.insert(self.y_splits[split], v['y'][{{(idx-1)*batch_size+1, idx*batch_size}}])
		end

		if extra ~= 0 then
			num_batches = num_batches + 1
			table.insert(self.x_splits[split], v['x'][{{num_examples-extra+1, num_examples}}])
			table.insert(self.y_splits[split], v['y'][{{num_examples-extra+1, num_examples}}])
		end
		self.split_sizes[split] = num_batches
	end
	print '<data init> finished data loading'
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
