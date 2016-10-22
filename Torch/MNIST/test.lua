require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'optim'

require 'LeNetModel'
require 'DataLoader'
require 'eval'

local utils = require 'utils'
local cmd = torch.CmdLine()

cmd:option('-checkpoint', 'cv/checkpoint_4000.t7')
local opt = cmd:parse(arg)

local dtype = 'torch.CudaTensor'
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

local loader = DataLoader(checkpoint.opt.batch_size)

model.evaluate()
local _, test_accuracy = eval('test')