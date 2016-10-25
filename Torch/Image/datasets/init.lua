--[[ code from https://github.com/facebook/fb.resnet.torch/blob/master/datasets/init.lua
--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet and CIFAR-10 datasets
--]]

local M = {}

local function isvalid(opt, cachePath)
	local imageInfo = torch.load(cachePath)
	if imageInfo.basedir and imageInfo.basedir ~= opt.data then
		return false
	end
	return true
end

function M.create(config)
	local cachePath = paths.concat(config.opt.gen, config.opt.dataset .. '.t7')
	if not paths.filep(cachePath) or not isvalid(config.opt, cachePath) then
		paths.mkdir('gen')

		local script = paths.dofile(config.opt.dataset .. '-gen.lua')
		script.exec(config.opt, cachePath)
	end
	local imageInfo = torch.load(cachePath)
	local Dataset = require('datasets/' .. config.opt.dataset)
	return Dataset(imageInfo, config)
end

return M
