local M = {}

function M.exec( opt, cacheFile )
	print("<data init> => Downloading MNIST dataset")
	local mnist = require 'mnist'
	local trainSet = mnist.traindataset()
	local testSet = mnist.testdataset()
	print("<data init> | saving MNIST dataset to " .. cacheFile)
	torch.save(cacheFile, {
		train = { data=trainSet.data[{{1, 50000}}]:contiguous():view(-1, 1, 28, 28), labels=trainSet.labels[{{1, 50000}}]:long():add(1):squeeze() },
		val = { data=trainSet.data[{{50001, 60000}}]:contiguous():view(-1, 1, 28, 28), labels=trainSet.labels[{{50001, 60000}}]:long():add(1):squeeze() },
		test = { data=testSet.data:contiguous():view(-1, 1, 28, 28), labels=testSet.labels:long():add(1):squeeze() }
	})
end