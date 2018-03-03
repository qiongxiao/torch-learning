--[[
--
--  code imitated https://github.com/facebook/fb.resnet.torch/blob/master/checkpoints.lua
--
--  checkpoints module
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

function checkpoint.loadCheckpointInfo(opt)
	if opt.resume == 'none' then
		return nil
	end

	local latestPath
	if opt.resumeType == 'best' then
		latestPath = paths.concat(opt.resume, 'best_info.t7')
	else
		latestPath = paths.concat(opt.resume, 'latest_info.t7')
	end
	if not paths.filep(latestPath) then
		error('<resuming> checkpoint' .. latestPath .. 'does not exist')
	end

	print('<resuming> => Loading checkpoint ' .. latestPath)
	local latest_info = torch.load(latestPath)
	local optimState
	if opt.resumeType == 'latest' then
		optimState = torch.load(paths.concat(opt.resume, latest_info.optimFile))
	end

	return latest_info, optimState
end

function checkpoint.saveModel(epoch, model, optimState, isBestModel, bestLoss, opt)
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

	local checkpointDir = paths.concat(opt.save, 'checkpoints')
	if not paths.dirp(checkpointDir) then
		os.execute('mkdir ' .. checkpointDir)
	end

	local cnnModelFile = 'model_cnn_' .. epoch .. '.t7'
	local seqModelFile = 'model_seq_' .. epoch .. '.t7'
	local optimFile = 'optimState_' .. epoch .. '.t7'

	torch.save(paths.concat(checkpointDir, cnnModelFile), cnn)
	torch.save(paths.concat(checkpointDir, seqModelFile), feature2seq)
	torch.save(paths.concat(checkpointDir, optimFile), optimState)
	torch.save(paths.concat(checkpointDir, 'latest_info.t7'), {
		epoch = epoch,
		cnnModelFile = cnnModelFile,
		seqModelFile = seqModelFile,
		optimFile = optimFile,
		bestLoss = bestLoss,
	})
	print('<checkpoint> => complete model saving')
	
	if isBestModel then
		print('<checkpoint> => begin best model saving')
		torch.save(paths.concat(checkpointDir, 'model_best_cnn.t7'), cnn)
		torch.save(paths.concat(checkpointDir, 'model_best_seq.t7'), feature2seq)
		torch.save(paths.concat(checkpointDir, 'best_info.t7'), {
		-- for finetuning from best model
		epoch = 1,
		cnnModelFile = 'model_best_cnn.t7',
		seqModelFile = 'model_best_seq.t7',
		bestLoss = bestLoss,
		bestEpoch = epoch,
	})
		print('<checkpoint> => complete best model saving')
	end
end

function checkpoint.cleanModel(epoch, opt)
	local checkpointDir = paths.concat(opt.save, 'checkpoints')

	local cnnModelFile = 'model_cnn_' .. epoch .. '.t7'
	local seqModelFile = 'model_seq_' .. epoch .. '.t7'
	local optimFile = 'optimState_' .. epoch .. '.t7'

	local cnnPath = paths.concat(checkpointDir, cnnModelFile)
	local seqPath = paths.concat(checkpointDir, seqModelFile)
	local optimPath = paths.concat(checkpointDir, optimFile)
	
	assert(paths.filep(cnnPath), 'Deleting file ' .. cnnPath .. 'not found')
	assert(paths.filep(seqPath), 'Deleting file ' .. seqPath .. 'not found')
	assert(paths.filep(optimPath), 'Deleting file ' .. optimPath .. 'not found')

	os.execute('rm ' .. cnnPath)
	os.execute('rm ' .. seqPath)
	os.execute('rm ' .. optimPath)
end

return checkpoint
