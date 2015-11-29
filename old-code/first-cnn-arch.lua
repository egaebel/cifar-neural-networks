require 'torch'
require 'nn'
require 'cifar10-data-loader'
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

--Normalize the data by subtracting the mean and dividing by the standard deviation
--data is a table with a 4D tensor at data, where color channel is 
--      the second dimension
--mean is an optional argument, it must be an array of means which will be used
--      to normalize the passed data instead of using its own
--stdv is an optional argument, it must be an array of standard deviations which
--      will be used to normalize the passed data instead of using its own
local function normalizeData(passedData, passedMean, passedStdv)
    mean = passedMean or {}
    stdv = passedStdv or {}
    for i = 1, 3 do-- loop over image channels
        if not passedMean then 
            mean[i] = passedData[{ {}, {i}, {}, {} }]:mean()
        end
        passedData[{ {}, {i}, {}, {} }]:add(-mean[i])

        if not passedStdv then
            stdv[i] = passedData[{ {}, {i}, {}, {}}]:std()
        end
        passedData[{ {}, {i}, {}, {} }]:div(stdv[i])
    end
    return mean, stdv
end

local function firstArch()
    net = nn.Sequential()
    --3 input channels, 6 output channels, 5x5 convolution kernel
    --1x1 strides, 3x3 padding
    net:add(nn.SpatialConvolution(3, 6, 5, 5, 1, 1, 3, 3))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    --6 input channels, 9 output channels, 5x5 convolution kernel
    --1x1 strides, 3x3 padding
    net:add(nn.SpatialConvolution(6, 9, 5, 5, 1, 1, 3, 3))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    --9 input channels, 12 output channels, 3x3 convolution kernel
    --1x1 strides, 2x2 padding
    net:add(nn.SpatialConvolution(9, 12, 3, 3, 1, 1, 2, 2))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    --12 input channels, 15 output channels, 3x3 convolution kernel
    --1x1 strides, 2x2 padding
    net:add(nn.SpatialConvolution(12, 15, 3, 3, 1, 1, 2, 2))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    --15 input channels, 18 output channels, 3x3 convolution kernel
    --1x1 strides, 2x2 padding
    net:add(nn.SpatialConvolution(15, 18, 3, 3, 1, 1, 2, 2))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    net:add(nn.View(18 * 2 * 2))
    net:add(nn.Linear(18 * 2 * 2, 120))
    net:add(nn.Linear(120, 84))
    net:add(nn.Linear(84, 10))
    net:add(nn.LogSoftMax()) -- Creates log-probability output    

    return net
end

--------------------------------------------------------------------------------
--Main Runner Code
--------------------------------------------------------------------------------
-- Setup data loader
local opt = {}
opt.batchSize = 10
opt.dataDir = 'torch-data'
local trainingDataLoader = Cifar10Loader(opt, 'train')
--local trainingDataset = trainingDataLoader.dataset
local trainMean, trainStdv = normalizeData(trainingDataLoader.data)

-- Train Net
net = firstArch()

-- Define loss function and stochastic gradient parameters and style
criterion = nn.ClassNLLCriterion()
-- CUDA-fy loss function and data set
if not EXCLUDE_CUDA_FLAG then
    criterion = criterion:cuda()
    dataset.data = dataset.data:cuda()
    net = net:cuda()
end
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 10
trainer:train(trainingDataLoader.data)

-- Validate data (eventually turn this into test)
local testingDataLoader = Cifar10Loader(opt, 'validate')
--local testingDataset = testingDataLoader.dataset
classes = testingDataLoader.classes
normalizeData(testingDataLoader.data, trainMean, trainStdv)
if not EXCLUDE_CUDA_FLAG then
    testingDataLoader.data = testingDataLoader.data:cuda()
end

-- Output predictions
predicted = net:forward(testingDataLoader.data[100])
predicted:exp()
for i = 1, predicted:size(1) do
    print(classes[i], predicted[i])
end

--Evaluate # correct
correct = 0
for i=1,10000 do
    local groundtruth = testingDataLoader.label[i]
    local prediction = net:forward(testingDataLoader.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

--Performance by class
print('\nPerformance by class')
class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i = 1, 10000 do
    local groundtruth = testingDataLoader.label[i]
    local prediction = net:forward(testingDataLoader.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end

for i = 1, #classes do
    print(classes[i], 100 * class_performance[i] / 1000 .. ' %')
end