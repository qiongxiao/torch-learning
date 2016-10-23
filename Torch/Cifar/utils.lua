local cjson = require 'cjson'

-- code from 'https://github.com/jcjohnson/torch-rnn/blob/master/util/utils.lua'
local utils = {}

function utils.write_json(path, obj)
	local f = io.open(path, 'w') 
	f:write(cjson.encode(obj))
	f:close()
end

function utils.get_kwarg(kwargs, name, default)
	if kwargs == nil then kwargs = {} end
	if kwargs[name] == nil and default == nil then
		assert(false, string.format('"%s" expected and not given', name))
	elseif kwargs[name] == nil then
		return default
	else
		return kwargs[name]
	end
end

-- code from 'https://github.com/joeyhng/trainplot'

local Plotter = torch.class('Plotter')

function Plotter:__init(path)
	self.path = path
	self.figures = {}
end

-- Information string displayed in the page
function Plotter:info(s)
	self.figures['info_str'] = s
end

function Plotter:add(fig_id, plot_id, iter, data)
	if data ~= data then data = -1 end
	if data == 1/0 then data = 1e-30 end
	if data == -1/0 then data = -1e-30 end

	if not fig_id then
		fig_id = plot_id
	end

	-- create figure if not exists
	if not self.figures[fig_id] then
		self.figures[fig_id] = {}
		self.figures[fig_id]['data'] = {}
		self.figures[fig_id]['layout'] = {['title']=fig_id}
	end

	local fig_data = self.figures[fig_id]['data']
	local plot
	for k, v in pairs(fig_data) do
		if v['name'] == plot_id then plot = v end
	end
	if not plot then
		plot = {['name'] = plot_id, ['x'] = {}, ['y'] = {}}
		table.insert(fig_data, plot)
	end
	table.insert(plot['x'], iter)
	table.insert(plot['y'], data)

	local file = io.open(self.path, 'w')
	file:write(cjson.encode(self.figures))
	file:close()
end

return utils
