require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'torchx'
local c = require 'trepl.colorize'
if itorch then
    path = require 'pl.path'
end

require 'cifar10-data-loader'

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

-- Got maybe like 75%?
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
    --net:add(nn.LogSoftMax()) -- Creates log-probability output    

    return net
end

-- Constructs and returns an inceptionModule from the paper 
-- "Going Deeper with Convolutional Networks", with input/output channels defined
-- with the parameters as follows:
-- inputChannels: the number of input channels
-- outputChannels: the expected number of outputChannels 
--                  (this parameter is only used to check the other parameters)
-- reductions: a 4-element array which specifies the number of channels output
--                  from each 1x1 convolutional network 
--                  (which should be smaller than the inputChannels usually...)
-- expansions: a 2-element array which specifies the number of channels output
--                  from the 3x3 convolutional layer and 
--                  the 5x5 convolutional layer
-- ReLU activations are applied after each convolutional layer
-- This module might be extended to allow for arbitrary width
local function inceptionModule(inputChannels, outputChannels, reductions, expansions)

    computedOutputChannels = reductions[1] + expansions[1] + expansions[2] + reductions[4]
    if not (outputChannels == computedOutputChannels) then
        print("\n\nOUTPUT CHANNELS DO NOT MATCH COMPUTED OUTPUT CHANNELS")
        print('outputChannels: ', outputChannels)
        print('computedOutputChannels: ', computedOutputChannels)
        print("\n\n")
        return nil
    end

    local inception = nn.DepthConcat(2)

    local column1 = nn.Sequential()
    column1:add(nn.SpatialConvolution(inputChannels, reductions[1],
        1, 1,  -- Convolution kernel
        1, 1)) -- Stride
    column1:add(nn.ReLU(true))
    inception:add(column1)
    
    local column2 = nn.Sequential()
    column2:add(nn.SpatialConvolution(inputChannels, reductions[2],
        1, 1, 
        1, 1))
    column2:add(nn.ReLU(true))
    column2:add(nn.SpatialConvolution(reductions[2], expansions[1],
        3, 3,  -- Convolution kernel
        1, 1)) -- Stride
    column2:add(nn.ReLU(true))
    inception:add(column2)

    local column3 = nn.Sequential()
    column3:add(nn.SpatialConvolution(inputChannels, reductions[3],
        1, 1, 
        1, 1))
    column3:add(nn.ReLU(true))
    column3:add(nn.SpatialConvolution(reductions[3], expansions[2],
        5, 5,  -- Convolution kernel
        1, 1)) -- Stride
    column3:add(nn.ReLU(true))
    inception:add(column3)

    local column4 = nn.Sequential()
    column4:add(nn.SpatialMaxPooling(3, 3, 1, 1))
    column4:add(nn.SpatialConvolution(inputChannels, reductions[4],
        1, 1,  -- Convolution kernel
        1, 1)) -- Stride
    column4:add(nn.ReLU(true))
    inception:add(column4)

    return inception
end

-- Achieved 80% on validation
-- Began to overfit
-- While achieving 80% on validation, achieved 90% on training
local function secondArch()
    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 64, 
        5, 5,
        1, 1))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.2))
    net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.SpatialConvolution(64, 128, 
        3, 3,
        2, 2))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.2))
    net:add(nn.SpatialBatchNormalization(128))
    -- Inception Module
    reductions = {
        64,
        64,
        32,
        128
    }
    expansions = {
        256,
        64
    }
    net:add(inceptionModule(128, 512, reductions, expansions))
    net:add(nn.SpatialConvolution(512, 768, 3, 3, 1, 1))
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
    -- Inception Module
    reductions = {
        64,
        256,
        256,
        128
    }
    expansions = {
        320,
        512
    }
    net:add(inceptionModule(768, 1024, reductions, expansions))
    net:add(nn.SpatialAveragePooling(5, 5, 1, 1))
    net:add(nn.View(1024))
    net:add(nn.Linear(1024, 512))
    net:add(nn.Dropout(0.4))
    net:add(nn.Linear(512, 256))
    net:add(nn.Dropout(0.4))
    net:add(nn.Linear(256, 10))


    return net
end

-- 75% test accuracy, so far.....
local function thirdArch()
    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 64, 
        5, 5,
        1, 1,
        2, 2))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.2))
    net:add(nn.SpatialMaxPooling(3, 3, 1, 1))
    net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.SpatialConvolution(64, 128, 
        3, 3,
        1, 1,
        1, 1))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.2))
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
    net:add(nn.SpatialBatchNormalization(128))
    -- Inception Module
    reductions = {
        32,
        32,
        16,
        32
    }
    expansions = {
        128,
        64
    }
    net:add(inceptionModule(128, 256, reductions, expansions))
    -- Inception Module
    reductions = {
        64,
        128,
        64,
        64
    }
    expansions = {
        256,
        128
    }
    net:add(inceptionModule(256, 512, reductions, expansions))
    net:add(nn.SpatialMaxPooling(3, 3, 1, 1))

    net:add(nn.SpatialConvolution(512, 768, 
            3, 3, 
            1, 1, 
            1, 1))
    net:add(nn.SpatialMaxPooling(3, 3, 1, 1))
    -- Inception Module
    reductions = {
        64,
        256,
        256,
        128
    }
    expansions = {
        320,
        512
    }
    net:add(inceptionModule(768, 1024, reductions, expansions))
    net:add(nn.SpatialAveragePooling(3, 3, 3, 3))
    net:add(nn.View(1024 * 3 * 3))
    net:add(nn.Linear(1024 * 3 * 3, 512))
    net:add(nn.Dropout(0.4))
    net:add(nn.Linear(512, 256))
    net:add(nn.Dropout(0.4))
    net:add(nn.Linear(256, 10))

    return net
end

--------------------------------------------------------------------------------
-- Main Runner Code-------------------------------------------------------------
--------------------------------------------------------------------------------
-- Setup data loader
local dataDir = 'torch-data'
local maxEpoch = 200
if EXCLUDE_CUDA_FLAG then
    sizeRestriction = 200
    maxEpoch = 5
end
local trainingDataLoader = Cifar10Loader(dataDir, 'train', sizeRestriction)
local opt = {
    netSaveDir = 'model-nets',
    batchSize = 128,
    learningRate = 1.0,
    weightDecay = 0.0005,
    momentum = 0.9,
    learningRateDecay = 0.00000001,
    maxEpoch = maxEpoch
}

-- Define neural network
net = thirdArch()
print("Network: ")
print(net)

-- Define loss function and stochastic gradient parameters and style
criterion = nn.CrossEntropyCriterion()

-- CUDA-fy loss function, model, and data set
if not EXCLUDE_CUDA_FLAG then
    criterion = criterion:cuda()
    trainingDataLoader:cuda()
    net = net:cuda()
end

-- Setup table for optimizer
optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay
}

-- Get NN parameters and set up confusion matrix
parameters, gradParameters = net:getParameters()
confusion = optim.ConfusionMatrix(trainingDataLoader.classes)

local function train()
    net:training()
    -- Setup mini-batches
    local indices = torch.randperm(trainingDataLoader.data:size(1)):long():split(opt.batchSize)
    indices[#indices] = nil
    local targets = torch.Tensor(opt.batchSize)
    if not EXCLUDE_CUDA_FLAG then
        targets = targets:cuda()
    end

    print('Training Beginning....')
    print('Training set size: ', trainingDataLoader.data:size(1))
    local tic = torch.tic()
    for t, v in ipairs(indices) do
        local inputs = trainingDataLoader.data:index(1, v)
        targets:copy(trainingDataLoader.labels:index(1, v))
        local feval = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end
            -- Zero out the gradient parameters from last iteration
            gradParameters:zero()
            
            local outputs = net:forward(inputs)
            local f = criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            net:backward(inputs, df_do)

            confusion:batchAdd(outputs, targets)

            return f, gradParameters
        end
        optim.sgd(feval, parameters, optimState)
    end
    confusion:updateValids()
    print(('Train accuracy: ' .. c.cyan'%.2f' .. ' %%\t time: %.2f s'):format(
            confusion.totalValid * 100, torch.toc(tic)))
    trainingAccuracy = confusion.totalValid * 100

    confusion:zero()
    print('------------------------------------------------')
end

local function test()
    -- Validate data (eventually turn this into test)
    local testingDataLoader = Cifar10Loader(dataDir, 'validate', sizeRestriction)
    local classes = testingDataLoader.classes
    if not EXCLUDE_CUDA_FLAG then
        testingDataLoader.data = testingDataLoader.data:cuda()
    end

    print('Beginning Testing')
    print('Testing set size: ', testingDataLoader.data:size(1))
    local tic = torch.tic()
    net:evaluate()
    for i = 1, testingDataLoader.data:size(1), opt.batchSize do
        local outputs
        if (i + opt.batchSize - 1) > testingDataLoader.data:size(1) then
            local endIndex = testingDataLoader.data:size(1) - i
            outputs = net:forward(testingDataLoader.data:narrow(1, i, endIndex))
            confusion:batchAdd(outputs, testingDataLoader.labels:narrow(1, i, endIndex))
        else
            outputs = net:forward(testingDataLoader.data:narrow(1, i, opt.batchSize))
            confusion:batchAdd(outputs, testingDataLoader.labels:narrow(1, i, opt.batchSize))
        end
    end

    confusion:updateValids()
    print(('Test Accuracy: ' .. c.cyan'%.2f' .. ' %%\t time: %.2f s'):format(
            confusion.totalValid * 100, torch.toc(tic)))
    print('------------------------------------------------')
end

--------------------------------------------------------------------------------
-- Runner code, main loop-------------------------------------------------------
--------------------------------------------------------------------------------
trainingDataLoader:groupAndSort()
--[[
print('Running for ', opt.maxEpoch, ' epochs')
local globalTic = torch.tic()
for i = 1, opt.maxEpoch do
    print('Epoch ', i)
    train()
    test()

    -- Visualize every 25 epochs
    if itorch and i % 25 == 0 then
        print('Visualizations: ')
        for j = 1, net:size() do
            print('\nLayer ', j)
            res = net:get(j).weight
            res = res:view(res:size(1), 1, res:size(2), res:size(3))
            itorch.image(res)
        end
    end

    -- Save model every 50 epochs
    if i % 50 == 0 then
        local filename = paths.concat(opt.netSaveDir, 'model.net')
        print('==> saving model to ' .. filename)
        torch.save(filename, net)
    end
end
print(('Total model trained in time: %f seconds'):format(torch.toc(globalTic)))
--]]