local t = require 'datasets/image_transform'

local M = {}

local MNISTDataset = torch.class('MNISTDataset', M)

function MNISTDataset:__init( imageInfo, config )
	local split = config.split or nil
	assert(imageInfo[split], split)
	self.imageInfo = imageInfo[split]
	self.config = config
end

function MNISTDataset:get( i )
	local image = self.imageInfo.data[i]:float()
	local label = self.imageInfo.labels[i]

	return {
		input = image,
		target = label,
	}
end

function MNISTDataset:size()
	return self.imageInfo.data:size(1)
end

-- Computed from entire MNIST training set (Excluding validation set)
local meanstd = {
	mean = 33.40,
	std = 78.67
}

function MNISTDataset:preprocess()
	if self.split == 'train' or self.split == 'val' or self.split == 'test' then
		return function(img)
			img = img:clone()
			img:csub(meanstd.mean)
			return img
		end
	else
		error('invalid split: ' .. self.split)
	end
end

return M.MNISTDataset