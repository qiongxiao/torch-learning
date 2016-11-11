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

function utils.findMaxScore(path)
	local figure = utils.readJson(path)
	local maxScore = {}
	for k, v in pairs(figure) do
		local name = v['layout']['title']
		local data = v['data']
		for s, t in pairs(data) do
			name = name .. " - " .. t['name']
			local smalldata = t['y']
			local max, maxid = 0, 0
			for u, w in pairs(smalldata) do
				if w > max then
					max = w
					maxid = u
				end
			end
			local maxt = {name = name, max = max, max_epoch = maxid}
			table.insert(maxScore, maxt)
		end
	end
	return maxScore
end

function utils.mergeFigures(paths)
	local finalFigure = {}
	for _, path in pairs(paths) do
		local filename = string.sub(path, 1, string.find(path, '.json')-1)
		local figure = utils.readJson(path)
		for title, subfig in pairs(figure) do
			if not finalFigure[title] then
				finalFigure[title] = subfig
			else
				local lines = subfig['data']
				for _, data in pairs(lines) do
					local name = filename .. ':' .. data['name']
					data['name'] = name
					table.insert(finalFigure[title]['data'], data)
				end
			end
		end
	end
	utils.writeJson('merge.json', finalFigure)
end

return utils
