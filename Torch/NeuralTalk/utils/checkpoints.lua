--[[
--
--  code from https://github.com/facebook/fb.resnet.torch/blob/master/checkpoints.lua
--
--  The training loop and learning rate schedule
--
--]]
local modelutils = require 'utils.modelutils'

local checkpoint = {}

local function deepCopy(tbl)
	-- creates a copy of a network with new modules and the same tensors
	local copy = {}
	for k, v in pairs(tbl) do
		if type(v) == 'table' then
			copy[k] = deepCopy(v)
		else
			copy[k] = v
		end
	end
	if torch.typename(tbl) then
		torch.setmetatable(copy, torch.typename(tbl))
	end
	return copy
end

function checkpoint.loadLatestInfo(opt)
	if opt.resume == 'none' then
		return nil
	end

	local latestPath = paths.concat(opt.resume, 'latest_info.t7')
	if not paths.filep(latestPath) then
		error('<resuming> checkpoint' .. latestPath .. 'does not exist')
	end

	print('<resuming> => Loading checkpoint ' .. latestPath)
	local latest_info = torch.load(latestPath)
	local optimState = torch.load(paths.concat(opt.resume, latest_info.optimFile))

	return latest_info, optimState
end

function checkpoint.saveModel(epoch, model, optimState, isBestModel, opt)
	-- create a clean copy on the CPU without modifying the original network
	print('<checkpoint> => begin cnn copy')
	local cnn = deepCopy(model.cnn):float():clearState()
	print('<checkpoint> => end cnn copy')
	print('<checkpoint> => begin feature2seq copy')
	-- cannot deepCopy because lstm has many clones of one lstmCell
	
	local feature2seq = modelutils.createFeature2seq(model.feature2seq)
	modelutils.cloneFeature2seq(feature2seq, model.feature2seq)
	feature2seq:float():clearState()
	print('<checkpoint> => end feature2seq copy')
	
	print('<checkpoint> => begin model saving')
	local cnnModelFile = 'model_cnn_' .. epoch .. '.t7'
	local seqModelFile = 'model_seq_' .. epoch .. '.t7'
	local optimFile = 'optimState_' .. epoch .. '.t7'
	torch.save(paths.concat(opt.save, cnnModelFile), cnn)
	torch.save(paths.concat(opt.save, seqModelFile), feature2seq)
	torch.save(paths.concat(opt.save, optimFile), optimState)
	torch.save(paths.concat(opt.save, 'latest_info.t7'), {
		epoch = epoch,
		cnnModelFile = cnnModelFile,
		seqModelFile = seqModelFile,
		optimFile = optimFile,
	})
	print('<checkpoint> => complete model saving')
	
	if isBestModel then
		print('<checkpoint> => begin best model saving')
		torch.save(paths.concat(opt.save, 'model_best_cnn.t7'), cnn)
		torch.save(paths.concat(opt.save, 'model_best_seq.t7'), feature2seq)
		print('<checkpoint> => complete best model saving')
	end
end

function checkpoint.cleanModel(epoch, opt)
	local cnnModelFile = 'model_cnn_' .. epoch .. '.t7'
	local seqModelFile = 'model_seq_' .. epoch .. '.t7'
	local optimFile = 'optimState_' .. epoch .. '.t7'
	local cnnPath = paths.concat(opt.save, cnnModelFile)
	local seqPath = paths.concat(opt.save, seqModelFile)
	local optimPath = paths.concat(opt.save, optimFile)
	
	assert(paths.filep(cnnModelFile), 'Deleting file' .. cnnModelFile .. 'not found')
	assert(paths.filep(seqModelFile), 'Deleting file' .. seqModelFile .. 'not found')
	assert(paths.filep(optimFile), 'Deleting file' .. optimFile .. 'not found')

	os.execute('rm ' .. cnnPath)
	os.execute('rm ' .. seqPath)
	os.execute('rm ' .. optimPath)
end

return checkpoint
