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
end
if pcall(cudaRequires) then
    print('Imported cuda modules in first-cnn-arch')
else
    print('Failed to import cuda modules in first-cnn-arch')
    EXCLUDE_CUDA_FLAG = true
end

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

local function loadModelFromFile(fileName)
    local filePath = paths.concat('model-nets', fileName)
    print('==> loading model from ' .. filePath)
    net = torch.load(filePath)
    return net
end

--------------------------------------------------------------------------------
---------------------------MODEL SELECTION--------------------------------------
--------------------------------------------------------------------------------
local LOAD_MODEL_FROM_FILE = true
local archOutputName = 'sixth-arch--layer-wise-test.net'
if LOAD_MODEL_FROM_FILE then
    _, opt = sixthArch()
    -- left off at epoch 13
    net = loadModelFromFile('sixth-arch--layer-wise-test.net_epoch_1')
else
    -- Define neural network
    net, opt = sixthArch()
end
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

    _, outputVector = torch.max(outputs, 2)

    for i = 1, targets:size(1) do
        if outputVector[i][1] == targets[i] then
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
                if epoch > 1 then
                    local numCorrect, numIncorrect = numCorrectIncorrect(outputs, targets)
                    -- When the right wrong threshold is crossed transform the
                    -- largest weights to be reduced
                    -- 1 - 0.20 = 0.80 (at 80% accuracy this kicks in)
                    local RIGHT_WRONG_THESHOLD = 0.2
                    local PERCENT_TOTAL_WEIGHT_CUTOFF = 0
                    if numIncorrect / numCorrect < RIGHT_WRONG_THESHOLD then
                        -- Allocate space on first runs, zero on subsequent runs
                        if not layerOptimConfig[i].learningRates then
                            -- Assign learning rates based on whether there was an error
                            layerOptimConfig[i].learningRates = torch.zeros(layerGradParameters[i]:size(1), 1)
                        else
                            layerOptimConfig[i].learningRates:zero()
                        end
                        local weightsSum = 0
                        for j = 1, layerParameters[i]:size(1) do
                            weightsSum = weightsSum + torch.abs(layerParameters[i][j])
                        end
                        local percentTotalWeight
                        for j = 1, layerParameters[i]:size(1) do
                            percentTotalWeight = (torch.abs(layerParameters[i][j]) / weightsSum) * 100
                            layerOptimConfig[i].learningRates[j] = (1 - percentTotalWeight)
                        end
                    else
                        -- Remove the learning rates if numIncorrect / numCorrect rises above threshold again
                        layerOptimConfig[i].learningRates = nil
                    end
                end
                -- Perform optimization
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
    --if i % 10 == 0 then
    --Changed for positive reinforcement
    local filename = paths.concat(opt.netSaveDir, archOutputName .. "_epoch_" .. i)
    print('==> saving model to ' .. filename .. "_epoch_" .. i)
    torch.save(filename, net)
    --end
end
print(('Total model trained in time: %f seconds'):format(torch.toc(globalTic)))
--]]