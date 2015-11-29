require 'torch'
require 'nn'
require 'SpatialBatchNormalization_nc'

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

-- Got maybe like 75%?
function firstArch()

    local opt = {
        netSaveDir = 'model-nets',
        batchSize = 128,
        learningRate = 1.0,
        weightDecay = 0.0005,
        momentum = 0.9,
        learningRateDecay = 0.00000001,
    }

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

    return net, opt
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
-- batchNormInceptionModule adds Batch Normalization before each non-linearity
local function batchNormInceptionModule(inputChannels, outputChannels, reductions, expansions)

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
    column1:add(nn.SpatialBatchNormalization_nc(reductions[1]))
    column1:add(nn.ReLU(true))
    inception:add(column1)
    
    local column2 = nn.Sequential()
    column2:add(nn.SpatialConvolution(inputChannels, reductions[2],
            1, 1, 
            1, 1))
    column2:add(nn.SpatialBatchNormalization_nc(reductions[2]))
    column2:add(nn.ReLU(true))
    column2:add(nn.SpatialConvolution(reductions[2], expansions[1],
            3, 3,  -- Convolution kernel
            1, 1)) -- Stride
    column2:add(nn.SpatialBatchNormalization_nc(expansions[1]))
    column2:add(nn.ReLU(true))
    inception:add(column2)

    local column3 = nn.Sequential()
    column3:add(nn.SpatialConvolution(inputChannels, reductions[3],
            1, 1, 
            1, 1))
    column3:add(nn.SpatialBatchNormalization_nc(reductions[3]))
    column3:add(nn.ReLU(true))
    column3:add(nn.SpatialConvolution(reductions[3], expansions[2],
            5, 5,  -- Convolution kernel
            1, 1)) -- Stride
    column3:add(nn.SpatialBatchNormalization_nc(expansions[2]))
    column3:add(nn.ReLU(true))
    inception:add(column3)

    local column4 = nn.Sequential()
    column4:add(nn.SpatialMaxPooling(3, 3, 1, 1))
    column4:add(nn.SpatialConvolution(inputChannels, reductions[4],
        1, 1,  -- Convolution kernel
        1, 1)) -- Stride
    column4:add(nn.SpatialBatchNormalization_nc(reductions[4]))
    column4:add(nn.ReLU(true))
    inception:add(column4)

    return inception
end

---------------------SGD TRAINING-----------------------------------------------
-- Achieved 81% on validation
-- Began to overfit
-- While achieving 81% on validation, achieved 90% on training
function secondArch()

    local opt = {
        netSaveDir = 'model-nets',
        batchSize = 128,
        learningRate = 1.0,
        weightDecay = 0.0005,
        momentum = 0.9,
        learningRateDecay = 0.00000001,
    }

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

    return net, opt
end

function secondArchNoNormalization()

    local opt = {
        netSaveDir = 'model-nets',
        batchSize = 128,
        learningRate = 1.0,
        weightDecay = 0.0005,
        momentum = 0.9,
        learningRateDecay = 0.00000001,
    }

    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 64, 
        5, 5,
        1, 1))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.2))
    net:add(nn.SpatialConvolution(64, 128, 
        3, 3,
        2, 2))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.2))
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

    return net, opt
end

---------------------SGD TRAINING-----------------------------------------------
--Obtained a maximum of 77% on validation set
function secondArchTuned()

    local opt = {
        netSaveDir = 'model-nets',
        batchSize = 128,
        learningRate = 1.0,
        weightDecay = 0.0005,
        momentum = 0.9,
        learningRateDecay = 0.00000001,
    }

    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 64, 
        5, 5,
        1, 1))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.2))
    net:add(nn.SpatialConvolution(64, 128, 
        3, 3,
        2, 2))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.2))
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
    -- Inception Module
    reductions = {
        128,
        512,
        512,
        256
    }
    expansions = {
        640,
        1024
    }
    net:add(inceptionModule(1024, 2048, reductions, expansions))
    net:add(nn.SpatialAveragePooling(5, 5, 1, 1))
    net:add(nn.View(2048))
    net:add(nn.Linear(2048, 1024))
    net:add(nn.Dropout(0.4))
    net:add(nn.Linear(1024, 512))
    net:add(nn.Dropout(0.4))
    net:add(nn.Linear(512, 10))

    return net, opt
end

---------------------SGD TRAINING-----------------------------------------------
-- 80% test accuracy
function thirdArch()

    local opt = {
        netSaveDir = 'model-nets',
        batchSize = 128,
        learningRate = 1.0,
        weightDecay = 0.0005,
        momentum = 0.9,
        learningRateDecay = 0.00000001,
    }

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

    return net, opt
end

function allInception()

    local opt = {
        netSaveDir = 'model-nets',
        batchSize = 128,
        learningRate = 1.0,
        weightDecay = 0.0005,
        momentum = 0.9,
        learningRateDecay = 0.00000001,
    }

    net = nn.Sequential()

    -- Inception Module
    reductions = {
        3,
        3,
        3,
        3
    }
    expansions = {
        5,
        5
    }
    net:add(inceptionModule(3, 16, reductions, expansions)) 
    -- Inception Module
    reductions = {
        4,
        8,
        4,
        4
    }
    expansions = {
        16,
        8
    }
    net:add(inceptionModule(16, 32, reductions, expansions)) 
    net:add(nn.SpatialMaxPooling(3, 3, 1, 1))

    -- Inception Module
    reductions = {
        8,
        16,
        8,
        8
    }
    expansions = {
        32,
        16
    }
    net:add(inceptionModule(32, 64, reductions, expansions)) 
    net:add(nn.SpatialMaxPooling(3, 3, 1, 1))
    -- Inception Module
    reductions = {
        16,
        32,
        16,
        16
    }
    expansions = {
        64,
        32
    }
    net:add(inceptionModule(64, 128, reductions, expansions)) 
    -- Inception Module
    reductions = {
        32,
        64,
        32,
        32
    }
    expansions = {
        128,
        64
    }
    net:add(inceptionModule(128, 256, reductions, expansions)) 
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2))

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
    net:add(nn.SpatialAveragePooling(3, 3, 3, 3))

    net:add(nn.View(512 * 4 * 4))
    net:add(nn.Linear(512 * 4 * 4, 256))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.4))
    net:add(nn.Linear(256, 10))

    return net, opt
end

function secondArchConvPadding()

    local opt = {
        netSaveDir = 'model-nets',
        batchSize = 128,
        learningRate = 1.0,
        weightDecay = 0.0005,
        momentum = 0.9,
        learningRateDecay = 0.00000001,
    }

    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 64, 
            5, 5,
            1, 1,
            2, 2))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.2))
    net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.SpatialMaxPooling(5, 5, 1, 1)) -- Equivalent to reduction from convolution in original
    net:add(nn.SpatialConvolution(64, 128, 
            3, 3,
            2, 2,
            17, 17))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.2))
    net:add(nn.SpatialBatchNormalization(128))
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2)) -- Equivalent to reduction from convolution in original
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
    net:add(nn.SpatialConvolution(512, 768, 
            3, 3, 
            1, 1,
            1, 1))
    net:add(nn.SpatialMaxPooling(3, 3, 1, 1)) -- Equivalent to reduction from convolution in original
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2)) -- Yes, it is odd to do two of these in a row....
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

    return net, opt
end

function secondArchConvPaddingNoFullyConnectedLayers()

    local opt = {
        netSaveDir = 'model-nets',
        batchSize = 128,
        learningRate = 1.0,
        weightDecay = 0.0005,
        momentum = 0.9,
        learningRateDecay = 0.00000001,
    }

    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 64, 
            5, 5,
            1, 1,
            2, 2))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.2))
    net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.SpatialMaxPooling(5, 5, 1, 1)) -- Equivalent to reduction from convolution in original
    net:add(nn.SpatialConvolution(64, 128, 
            3, 3,
            2, 2,
            17, 17))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.2))
    net:add(nn.SpatialBatchNormalization(128))
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2)) -- Equivalent to reduction from convolution in original
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
    net:add(nn.SpatialConvolution(512, 768, 
            3, 3, 
            1, 1,
            1, 1))
    net:add(nn.SpatialMaxPooling(3, 3, 1, 1)) -- Equivalent to reduction from convolution in original
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2)) -- Yes, it is odd to do two of these in a row....
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

    -- Downsample
    net:add(nn.SpatialConvolution(1024, 10, 
            1, 1, 
            1, 1))
    net:add(nn.SpatialAveragePooling(3, 3, 3, 3))
    net:add(nn.Reshape(10))

    return net, opt
end

function secondArchConvPaddingNoFullyConnectedLayersHeavyDropout()

    local opt = {
        netSaveDir = 'model-nets',
        batchSize = 128,
        learningRate = 1.0,
        weightDecay = 0.0005,
        momentum = 0.9,
        learningRateDecay = 0.00000001,
    }

    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 64, 
            5, 5,
            1, 1,
            2, 2))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.4))
    net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.SpatialMaxPooling(5, 5, 1, 1)) -- Equivalent to reduction from convolution in original
    net:add(nn.SpatialConvolution(64, 128, 
            3, 3,
            2, 2,
            17, 17))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.4))
    net:add(nn.SpatialBatchNormalization(128))
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2)) -- Equivalent to reduction from convolution in original
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
    net:add(nn.Dropout(0.4))
    net:add(nn.SpatialConvolution(512, 768, 
            3, 3, 
            1, 1,
            1, 1))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.4))
    net:add(nn.SpatialMaxPooling(3, 3, 1, 1)) -- Equivalent to reduction from convolution in original
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2)) -- Yes, it is odd to do two of these in a row....
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
    net:add(nn.Dropout(0.4))

    -- Downsample
    net:add(nn.SpatialConvolution(1024, 10, 
            1, 1, 
            1, 1))
    net:add(nn.Dropout(0.4))
    net:add(nn.SpatialAveragePooling(3, 3, 3, 3))
    net:add(nn.Reshape(10))

    return net, opt
end

function secondArchConvPaddingNoFullyConnectedLayersBetterNormalization()

    local opt = {
        netSaveDir = 'model-nets',
        batchSize = 128,
        learningRate = 1.0,
        weightDecay = 0.0005,
        momentum = 0.9,
        learningRateDecay = 0.00000001,
    }

    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 64, 
            5, 5,
            1, 1,
            2, 2))
    net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(5, 5, 1, 1)) -- Equivalent to reduction from convolution in original
    net:add(nn.SpatialConvolution(64, 128, 
            3, 3,
            2, 2,
            17, 17))
    net:add(nn.SpatialBatchNormalization(128))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2)) -- Equivalent to reduction from convolution in original
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
    net:add(nn.SpatialConvolution(512, 768, 
            3, 3, 
            1, 1,
            1, 1))
    net:add(nn.SpatialMaxPooling(3, 3, 1, 1)) -- Equivalent to reduction from convolution in original
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2)) -- Yes, it is odd to do two of these in a row....
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

    -- Downsample
    net:add(nn.SpatialConvolution(1024, 10, 
            1, 1, 
            1, 1))
    net:add(nn.SpatialAveragePooling(3, 3, 3, 3))
    net:add(nn.Reshape(10))

    return net, opt
end

function secondArchConvPaddingNoFullyConnectedLayersBetterNormalizationFullConvNorm()

    local opt = {
        netSaveDir = 'model-nets',
        batchSize = 128,
        learningRate = 1.0,
        weightDecay = 0.0005,
        momentum = 0.9,
        learningRateDecay = 0.00000001,
    }

    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 64, 
            5, 5,
            1, 1,
            2, 2))
    net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(5, 5, 1, 1)) -- Equivalent to reduction from convolution in original
    net:add(nn.SpatialConvolution(64, 128, 
            3, 3,
            2, 2,
            17, 17))
    net:add(nn.SpatialBatchNormalization(128))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2)) -- Equivalent to reduction from convolution in original
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
    net:add(nn.SpatialConvolution(512, 768, 
            3, 3, 
            1, 1,
            1, 1))
    net:add(nn.SpatialBatchNormalization(768))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(3, 3, 1, 1)) -- Equivalent to reduction from convolution in original
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2)) -- Yes, it is odd to do two of these in a row....
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

    -- Downsample
    net:add(nn.SpatialConvolution(1024, 10, 
            1, 1, 
            1, 1))
    net:add(nn.SpatialBatchNormalization(10))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialAveragePooling(3, 3, 3, 3))
    net:add(nn.Reshape(10))

    return net, opt
end

function secondArchConvPaddingNoFullyConnectedLayersBetterNormalizationNormedInception()

    local opt = {
        netSaveDir = 'model-nets',
        batchSize = 128,
        learningRate = 1.0,
        weightDecay = 0.0005,
        momentum = 0.9,
        learningRateDecay = 0.00000001,
    }

    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 64, 
            5, 5,
            1, 1,
            2, 2))
    net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(5, 5, 1, 1)) -- Equivalent to reduction from convolution in original
    net:add(nn.SpatialConvolution(64, 128, 
            3, 3,
            2, 2,
            17, 17))
    net:add(nn.SpatialBatchNormalization(128))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2)) -- Equivalent to reduction from convolution in original
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
    net:add(batchNormInceptionModule(128, 512, reductions, expansions))
    net:add(nn.SpatialConvolution(512, 768, 
            3, 3, 
            1, 1,
            1, 1))
    net:add(nn.SpatialMaxPooling(3, 3, 1, 1)) -- Equivalent to reduction from convolution in original
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2)) -- Yes, it is odd to do two of these in a row....
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
    net:add(batchNormInceptionModule(768, 1024, reductions, expansions))

    -- Downsample
    net:add(nn.SpatialConvolution(1024, 10, 
            1, 1, 
            1, 1))
    net:add(nn.SpatialAveragePooling(3, 3, 3, 3))
    net:add(nn.Reshape(10))

    return net, opt
end


-- No inception
function fifthArch()

    local opt = {
        netSaveDir = 'model-nets',
        batchSize = 128,
        learningRate = 1.0,
        weightDecay = 0.0005,
        momentum = 0.9,
        learningRateDecay = 0.00000001,
    }

    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 64, 
            3, 3,
            1, 1,
            1, 1))
    net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.2))
    net:add(nn.SpatialMaxPooling(3, 3, 1, 1))

    net:add(nn.SpatialConvolution(64, 128, 
            3, 3,
            1, 1,
            1, 1))
    net:add(nn.SpatialBatchNormalization(128))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.2))
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2))

    net:add(nn.SpatialConvolution(128, 512,
            3, 3,
            1, 1,
            1, 1))
    net:add(nn.SpatialBatchNormalization(512))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(3, 3, 1, 1))
    net:add(nn.SpatialConvolution(512, 768, 
            3, 3, 
            1, 1,
            1, 1))
    net:add(nn.SpatialBatchNormalization(768))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2))

    net:add(nn.SpatialConvolution(768, 1024,
            3, 3,
            1, 1,
            1, 1))
    net:add(nn.SpatialBatchNormalization(1024))
    net:add(nn.ReLU(true))

    -- Downsample
    net:add(nn.SpatialConvolution(1024, 10, 
            1, 1, 
            1, 1))
    net:add(nn.SpatialAveragePooling(3, 3, 3, 3))
    net:add(nn.Reshape(10))

    return net, opt
end

-- No inception
-- Fixed dropout...
function sixthArch()

    local opt = {
        netSaveDir = 'model-nets',
        batchSize = 128,
        learningRate = 1.0,
        weightDecay = 0.0005,
        momentum = 0.9,
        learningRateDecay = 0.00000001,
    }

    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 64, 
            3, 3,
            1, 1,
            1, 1))
    net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(3, 3, 1, 1))

    net:add(nn.SpatialDropout(0.2))
    net:add(nn.SpatialConvolution(64, 128, 
            3, 3,
            1, 1,
            1, 1))
    net:add(nn.SpatialBatchNormalization(128))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2))

    net:add(nn.SpatialDropout(0.2))
    net:add(nn.SpatialConvolution(128, 512,
            3, 3,
            1, 1,
            1, 1))
    net:add(nn.SpatialBatchNormalization(512))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(3, 3, 1, 1))
    net:add(nn.SpatialConvolution(512, 768, 
            3, 3, 
            1, 1,
            1, 1))
    net:add(nn.SpatialBatchNormalization(768))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2))

    net:add(nn.SpatialConvolution(768, 1024,
            3, 3,
            1, 1,
            1, 1))
    net:add(nn.SpatialBatchNormalization(1024))
    net:add(nn.ReLU(true))

    -- Downsample
    net:add(nn.SpatialConvolution(1024, 10, 
            1, 1, 
            1, 1))
    net:add(nn.SpatialAveragePooling(3, 3, 3, 3))
    net:add(nn.Reshape(10))

    return net, opt
end

function layerWiseTest() 
    local opt = {
        netSaveDir = 'model-nets',
        batchSize = 128,
        learningRate = 1.0,
        weightDecay = 0.0005,
        momentum = 0.9,
        learningRateDecay = 0.00000001,
    }

    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1))
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
    net:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
    net:add(nn.SpatialMaxPooling(3, 3, 3, 3))
    net:add(nn.SpatialConvolution(128, 10, 1, 1, 1, 1))
    net:add(nn.SpatialAveragePooling(3, 3, 3, 3))
    net:add(nn.Reshape(10))

    return net, opt    
end