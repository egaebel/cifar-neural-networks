require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'models'
local c = require 'trepl.colorize'
if itorch then
    path = require 'pl.path'
end

require 'cifar10-data-loader'

-- CUDA Import Stuff
local EXCLUDE_CUDA_FLAG = false
local function cudaRequires() 
    print('Importing cutorch....')
    require 'cutorch'
    print('Importing cunn.......')
    require 'cunn'
    --print('Importing cudnn......')
    --require 'cudnn'
end
if pcall(cudaRequires) then
    print('Imported cuda modules in first-cnn-arch')
else
    print('Failed to import cuda modules in first-cnn-arch')
    EXCLUDE_CUDA_FLAG = true
end
--[[
local nnLib = nn
if not EXCLUDE_CUDA_FLAG then
    nnLib = cudnn
end
--]]



--------------------------------------------------------------------------------
-- Main Runner Code-------------------------------------------------------------
--------------------------------------------------------------------------------
-- Setup data loader
local dataDir = 'torch-data'
local trainingDataLoader = Cifar10Loader(dataDir, 'train', sizeRestriction)
local testingDataLoader = Cifar10Loader(dataDir, 'validate', sizeRestriction)
--local testingDataLoader = Cifar10Loader(dataDir, 'test', sizeRestriction)
local classes = testingDataLoader.classes

--------------------------------------------------------------------------------
---------------------------MODEL SELECTION--------------------------------------
--------------------------------------------------------------------------------
-- Define neural network
--local archOutputName = 'second-arch-conv-padding-no-fully-connected-better-normalization-normed-inception.net'
--net, opt = secondArchConvPaddingNoFullyConnectedLayersBetterNormalizationNormedInception()
local archOutputName = 'sixth-arch-more-dropout.net'
net, opt = sixthArchMoreDropout()
print("Network: ")
print(net)
print("Options: ")
print(opt)
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
local maxEpoch = 200
if EXCLUDE_CUDA_FLAG then
    sizeRestriction = 200
    maxEpoch = 5
end
opt.maxEpoch = maxEpoch


-- Define loss function and stochastic gradient parameters and style
criterion = nn.CrossEntropyCriterion()

-- CUDA-fy loss function, model, and data set
if not EXCLUDE_CUDA_FLAG then
    criterion = criterion:cuda()
    trainingDataLoader:cuda()
    testingDataLoader:cuda()
    net = net:cuda()
end

-- Setup table for optimizer parameters
optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay
}

-- Get NN parameters and set up confusion matrix
local parameters, gradParameters = net:getParameters()
local confusion = optim.ConfusionMatrix(trainingDataLoader.classes)

local function train()
    collectgarbage()
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
        collectgarbage()
        optim.sgd(feval, parameters, optimState)
    end
    confusion:updateValids()
    print(('Train accuracy: ' .. c.cyan'%.2f' .. ' %%\t time: %.2f s'):format(
            confusion.totalValid * 100, torch.toc(tic)))
    trainingAccuracy = confusion.totalValid * 100
    
    print('------------------------------------------------')
    print('Confusion Matrix:')
    print(confusion)
    confusion:zero()
    print('------------------------------------------------')
end

local function test()
    -- Validate data (eventually turn this into test)
    collectgarbage()
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
    print('Confusion Matrix:')
    print(confusion)
    confusion:zero()
    print('------------------------------------------------')
end

--------------------------------------------------------------------------------
-- Runner code, main loop-------------------------------------------------------
--------------------------------------------------------------------------------
collectgarbage()
print('Running for ', opt.maxEpoch, ' epochs')
local globalTic = torch.tic()
for i = 1, opt.maxEpoch do
    print('Epoch ', i)
    train()
    test()

    -- Save model every 10 epochs
    if i % 10 == 0 then
        local filename = paths.concat(opt.netSaveDir, archOutputName)
        print('==> saving model to ' .. filename)
        torch.save(filename, net)
    end
end
print(('Total model trained in time: %f seconds'):format(torch.toc(globalTic)))
--]]