--[[
--
--  code from https://github.com/jcjohnson/torch-rnn
--
--]]

local cjson = require 'cjson'

local utils = {}

function utils.readJson(path)
  local f = io.open(path, 'r')
  local s = f:read('*all')
  f:close()
  return cjson.decode(s)
end

function utils.writeJson(path, obj)
	local f = io.open(path, 'w') 
	f:write(cjson.encode(obj))
	f:close()
end

return utils
