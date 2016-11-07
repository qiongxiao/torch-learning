local netutils = {}

function netutils.convInit(model, name)
	for k,v in pairs(model:findModules(name)) do
		local n = v.kW*v.kH*v.nOutputPlane
		v.weight:normal(0,math.sqrt(2/n))
		if cudnn.version >= 4000 then
			v.bias = nil
			v.gradBias = nil
		else
			v.bias:zero()
		end
	end
end

function netutils.linearInit(model)
	for k,v in pairs(model:findModules('nn.Linear')) do
		local n = v.weight:size(2)+v.weight:size(1)
		v.weight:normal(0,math.sqrt(2/n))
		v.bias:zero()
	end
end

function netutils.zeroPartialParams(params, p)
	p = p or 0.5
	for _, v in pairs(params) do
		local vv = v:view(-1)
		local vvabs = torch.abs(vv)
		local num = vv:size(1)
		local _, vk = torch.topk(vvabs, math.floor(num*p))
		vv:indexFill(1, vk, 0)
	end
end

return netutils