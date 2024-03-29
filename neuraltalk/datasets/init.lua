--[[
--
--  code from https://github.com/facebook/fb.resnet.torch/blob/master/datasets/init.lua
--
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
	local cachePath = paths.concat('gen', config.opt.dataset .. '.t7')
	if not paths.filep(cachePath) or not isvalid(config.opt, cachePath) then
		paths.mkdir('gen')

		local script = paths.dofile(config.opt.dataset .. '_gen.lua')
		script.exec(config.opt, cachePath)
	end
	local imageInfo = torch.load(cachePath)
	local Dataset = require('datasets/' .. config.opt.dataset)
	return Dataset(imageInfo, config.opt, config.split)
end

return M
