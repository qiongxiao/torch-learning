--[[
--
--  code adaption from https://github.com/joeyhng/trainplot
--
--]]
require 'paths'
local utils = require 'utils.utils'

local M = {}
local Plotter = torch.class('Plotter', M)

function Plotter:__init(opt, checkpoint)
	self.path = paths.concat(opt.save, 'plot/out.json')
	self.figures = {}
	self.checkpoint_path = paths.concat(opt.save, 'plot/out_checkpoint.json')
	print('<plot init> loading plot')
	if (not paths.filep(self.path)) then
		paths.mkdir(paths.dirname(self.path))
		self.figures['info_str'] = {created_time=io.popen('date'):read(), tag='Plot'}
	end
	if checkpoint then
		local startEpoch = checkpoint.epoch
		if paths.filep(self.checkpoint_path) then
			self.figures = utils.readJson(self.checkpoint_path)
			print('<plot init> correcting loading plot')
			for name, fig in pairs(self.figures) do
				local data = fig['data']
				if name ~= 'Train Loss - Iteration' then
					for _, plot in pairs(data) do
						local y = plot['y']
						local x = plot['x']
						for k, v in pairs(y) do
							if k > startEpoch then
								y[k] = nil
							end
						end
						for k, v in pairs(x) do
							if k > startEpoch then
								x[k] = nil
							end
						end
					end
				end
			end
			print('<plot init> finishing loading plot')
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

function Plotter:checkpoint()
	utils.writeJson(self.checkpoint_path, self.figures)
end

return M.Plotter
