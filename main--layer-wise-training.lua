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
local maxEpoch = 200
if EXCLUDE_CUDA_FLAG then
    sizeRestriction = 200
    maxEpoch = 5
end

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
local archOutputName = 'sixth-arch--layer-wise-test.net'
net, opt = sixthArch()
print("Network: ")
print(net)
print("Options: ")
print(opt)
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
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
local memoryLength = opt.batchSize
optimConfig = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay
    -- Learning rates and related parameters
    -- Start off with equalized right/wrong ratio
    --[[
    memoryLength = memoryLength,
    numWrong = memoryLength / 2,
    numRight = memoryLength / 2,
    learningRates = nil
    --]]
}
local function copyTable(mytable)
    newtable = {}
    for k, v in pairs(mytable) do
        newtable[k] = v
    end
    return newtable
end

------------------------------------------------------------------
-- The basic goal of this separate main.lua file is the following
-- Use sgd to train on each layer of the net
-- this requires me to get separate...EVERYTHING for each layer
-- separate parameters, separate gradParameters, etc.
------------------------------------------------------------------

-- Get NN parameters and set up confusion matrix
parameters, gradParameters = net:getParameters()
-- Get layer-wise parameters and gradParameters
local layerOptimConfig = {}
local layerParameters = {}
local layerGradParameters = {}
for i = 1, net:size() do
    layer = net:get(i)
    local parameters, gradParameters = layer:getParameters()
    table.insert(layerParameters, parameters)
    table.insert(layerGradParameters, gradParameters)
    -- Create copies of optimConfig so they can be altered on the fly later
    table.insert(layerOptimConfig, copyTable(optimConfig))
end

local confusion = optim.ConfusionMatrix(trainingDataLoader.classes)

-- Takes outputs and targets, returns the number of correct and incorrect instances
local function numCorrectIncorrect(outputs, targets)
    local nCorrect = 0
    local nIncorrect = 0
    print("targets:size(): ")
    print(targets:size(1))
    for i = 1, targets:size(1) do
        _, output = torch.max(outputs, 2)
        if output == targets[i] then
            nCorrect = nCorrect + 1
        else
            nIncorrect = nIncorrect + 1
        end
    end
    return nCorrect, nIncorrect
end

-- Train the net, layer by layer, using adaptive learning rates
local function train()
    if not epoch then
        epoch = 1
    else
        epoch = epoch + 1
    end
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

        -- Zero (I'm actually not sure why this has to be done 
        --  on each layerGradParameter..there's something messed up with 
        --  references I think...)
        gradParameters:zero()
        for i = 1, #layerGradParameters do
            layerGradParameters[i]:zero()
        end

        -- Run network forward and backward
        local outputs = net:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        net:backward(inputs, df_do)
        confusion:batchAdd(outputs, targets)

        -- Perform layer-wise optimization on the weights
        for i = 1, #layerParameters do
            -- If this layer has parameters to be optimized
            if layerParameters[i]:nDimension() ~= 0 then
                -- If this isn't the first epoch (we need right/wrong data first)
                --[[
                if epoch > 1 then
                    -- Allocate space on first runs, zero on subsequent runs
                    if not layerOptimConfig[i].learningRates then
                        -- Assign learning rates based on whether there was an error
                        layerOptimConfig[i].learningRates = torch.zeros(layerGradParameters[i]:size(1), 1)
                    else
                        layerOptimConfig[i].learningRates:zero()
                    end
                    layerOptimConfig[i].numRight numCorrectIncorrect(outputs, targets)
                    -- Sum up layerGradParameters
                    -- Divide each neuron layerGrad by the sum
                    -- Use this fraction to decide whether correct/incorrect applies

                end
                --]]

                local feval = function(x)
                    return _, layerGradParameters[i]
                end
                optim.sgd(feval, layerParameters[i], layerOptimConfig[i])
            end
        end
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

--[[

        print("Adding to layerOptimConfig....")
                -- If there were enough errors to be "significant"
                -- Sum up the number correct/incorrect in the batch
                -- If the number of correct or incorrect is "significant" update
                -- right/wrong ratio for the layer
                local nCorrect, nIncorrect = numCorrectIncorrect(outputs, targets)
                -- Increment the number right if at least a quarter of the batch was
                -- correct
                if nCorrect > opt.batchSize / 4 then
                    layerOptimConfig[i].numRight = layerOptimConfig[i].numRight + 1
                    layerOptimConfig[i].numWrong = layerOptimConfig[i].numWrong - 1
                -- Only increment the number wrong if there are lots right
                -- For now I'm only going to do positive reinforcement
                -- That is, the wrong/right ratio is <= 1
                elseif nIncorrect > opt.batchSize / 4 
                        and layerOptimConfig[i].numWrong < layerOptimConfig[i].numRight then
                    layerOptimConfig[i].numWrong = layerOptimConfig[i].numWrong + 1
                    layerOptimConfig[i].numRight = layerOptimConfig[i].numRight - 1
                end
                local rightWrongRatio = layerOptimConfig[i].numWrong / layerOptimConfig[i].numRight
                layerOptimConfig[i].learningRates:add(rightWrongRatio)


--]]

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