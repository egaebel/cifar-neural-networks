require 'torch'
require 'nn'
-- CUDA Import Stuff
local EXCLUDE_CUDA_FLAG = false
local function cudaRequires() 
    require 'cutorch'
    require 'cunn'
end
if pcall(cudaRequires) then
    print('Imported cuda modules in first-cnn-arch')
else
    print('Failed to import cuda modules in first-cnn-arch')
    EXCLUDE_CUDA_FLAG = true
end

local filename = paths.concat('model-nets', 'model.net')
print('==> loading model from ' .. filename)
net = torch.load(filename)
print(net)