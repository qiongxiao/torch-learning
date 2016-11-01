--[[
--
--  code from https://github.com/joeyhng/trainplot
--
--]]
local utils = require 'utils/utils'

local M = {}
local Plotter = torch.class('Plotter', M)

function Plotter:__init(opt)
	self.path = opt.plotPath .. '.json'
	self.figures = {}
	self.checkpoint_path = opt.plotPath .. '_checkpoint.json'
	print('<plot init> loading plot')
	if (not paths.filep(self.path)) then
		paths.mkdir(paths.dirname(self.path))
		self.figures['info_str'] = {created_time=io.popen('date'):read(), tag='Plot'}
	end
	if opt.resume ~= 'none' then
		if paths.filep(self.checkpoint_path) then
			self.figures = utils.readJson(self.checkpoint_path)
			print('<plot init> fininsh loading plot')
		else
			error(string.format('"%s" does not existed', self.checkpoint_path))
		end
	end
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

	utils.writeJson(self.path, self.figures)
end

function Plotter:save()
	utils.writeJson(self.checkpoint_path, self.figures)
end

return M.Plotter
