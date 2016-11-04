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

function utils.readTxt(path)
	local lines = {}
	for line in io.lines(path) do
		lines[#lines+1] = line
	end
	return lines
end

-- code from http://stackoverflow.com/questions/15706270/sort-a-table-in-lua
function utils.spairs(t, order)
	local keys = {}
	for k in pairs(t) do keys[#keys+1] = k end

	if order then
		table.sort(keys, function(a,b) return order(t, a, b) end)
	else
		table.sort(keys)
	end

	local i = 0
	return function()
		i = i + 1
		if keys[i] then
			return keys[i], t[keys[i]]
		end
	end
end

return utils