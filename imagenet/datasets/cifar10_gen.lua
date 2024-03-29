--[[
--
--  code from https://github.com/facebook/fb.resnet.torch/blob/master/datasets/cifar10-gen.lua
--
--  This automatically downloads the CIFAR-10 dataset from
--  http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz
--
--]]

local URL = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'

local M = {}

local function convertToTensor(files)
	local data, labels

	for _, file in ipairs(files) do
		local m = torch.load(file, 'ascii')
		if not data then
			data = m.data:t()
			labels = m.labels:squeeze()
		else
			data = torch.cat(data, m.data:t(), 1)
			labels = torch.cat(labels, m.labels:squeeze())
		end
	end

	-- This is *very* important. The downloaded files have labels 0-9, which do
	-- not work with CrossEntropyCriterion
	labels:add(1)

	return {
		data = data:contiguous():view(-1, 3, 32, 32),
		labels = labels,
	}
end

function M.exec(opt, cacheFile)
	print("<data init> => Downloading CIFAR-10 dataset from " .. URL)
	local ok = os.execute('curl ' .. URL .. ' | tar xz -C gen/')
	assert(ok == true or ok == 0, 'error downloading CIFAR-10')

	print("<data init> | combining dataset into a single file")
	local trainData = convertToTensor({
		'gen/cifar-10-batches-t7/data_batch_1.t7',
		'gen/cifar-10-batches-t7/data_batch_2.t7',
		'gen/cifar-10-batches-t7/data_batch_3.t7',
		'gen/cifar-10-batches-t7/data_batch_4.t7',
		'gen/cifar-10-batches-t7/data_batch_5.t7',
	})
	local testData = convertToTensor({
		'gen/cifar-10-batches-t7/test_batch.t7',
	})

	print("<data init> | saving CIFAR-10 dataset to " .. cacheFile)
	torch.save(cacheFile, {
		train = trainData,
		val = testData,
	})
end

return M
