require 'torch'
local EXCLUDE_CUDA_FLAG = false
local function importCutorch() 
    require 'cutorch'
end
if pcall(importCutorch) then
    print('Impored cutorch in cifar10-data-loader')
else
    print('Failed to import cutoch in cifar10-data-loader')
    EXCLUDE_CUDA_FLAG = true
end

Cifar10Loader = {}

local trainFile = 'cifar10-train.t7'
local validateFile = 'cifar10-validate.t7'
local testFile = 'cifar10-test.t7'
--------------------------------------------------------------------------------
--Below are non-class functions to download and convert the cifar10 data
--------------------------------------------------------------------------------

local function convertCifarBinToTorchTensor(inputFnames)
   local nSamples = 0
   --Iterate over files to get sizes
   for i = 1,#inputFnames do
      local inputFname = inputFnames[i]
      local m = torch.DiskFile(inputFname, 'r'):binary()
      m:seekEnd()
      local length = m:position() - 1
      local nSamplesF = length / 3073 -- 1 label byte, 3072 pixel bytes
      assert(nSamplesF == math.floor(nSamplesF), 'expecting numSamples to be an exact integer')
      nSamples = nSamples + nSamplesF
      m:close()
   end

   local label = torch.ByteTensor(nSamples)
   local data = torch.ByteTensor(nSamples, 3, 32, 32)

   --Iterate over files to fill ByteTensors
   local index = 1
   for i = 1,#inputFnames do
      local inputFname = inputFnames[i]
      local m = torch.DiskFile(inputFname, 'r'):binary()
      m:seekEnd()
      local length = m:position() - 1
      local nSamplesF = length / 3073 -- 1 label byte, 3072 pixel bytes
      m:seek(1)
      for j=1,nSamplesF do
         label[index] = m:readByte()
         local store = m:readByte(3072)
         data[index]:copy(torch.ByteTensor(store))
         index = index + 1
      end
      m:close()
   end

   local out = {}
   out.data = data
   out.label = label
   return out
end

local function prepCifar10Data(pathForData)
    pathForData = pathForData .. '/'
    --Testing parameters (can be used as sanity checks later)
    forceLoad = false
    forceConvert = false
    --Download and/or untar if necessary
    if not path.exists('cifar-10-binary.tar.gz') or forceLoad then
        print 'path cifar-10-binary.tar.gz DNE, downloading and extractin'
        os.execute('wget -c http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz')
        os.execute('tar -xvf cifar-10-binary.tar.gz')
    elseif not path.exists('cifar-10-batches-bin') then
        print 'path cifar-10-batches-bin DNE, extracting from cifar-10-binary.tar.gz'
        os.execute('tar -xvf cifar-10-binary.tar.gz')
    else
        print 'All base files and directories present, skipping to check format'
    end
    --Create folder for data being converted to be stored in
    if not paths.dir(pathForData) then
        paths.mkdir(pathForData)
    end
    --Convert training, validation, and test data from binary to torch tensor if necessary
    if not path.exists(pathForData .. trainFile) or forceConvert then
        --Set percentage to set aside for validation
        local validationRatio = .2
        --Load all training data
        local inputTrainFileNames = {'cifar-10-batches-bin/data_batch_1.bin',
            'cifar-10-batches-bin/data_batch_2.bin',
            'cifar-10-batches-bin/data_batch_3.bin',
            'cifar-10-batches-bin/data_batch_4.bin',
            'cifar-10-batches-bin/data_batch_5.bin'}
        local outputTrainFileName = pathForData .. trainFile
        local outputValidationFileName = pathForData .. validateFile
        local out = convertCifarBinToTorchTensor(inputTrainFileNames)
        local dataSize = out.label:size()[1]
        --Split training data into training/validation
        local numValidationSamples = dataSize * validationRatio
        local numTrainingSamples = dataSize - numValidationSamples
        local trainLabel = out.label:sub(1,numTrainingSamples)
        local trainData = out.data:sub(1,numTrainingSamples)
        local validateLabel = out.label:sub(numTrainingSamples, dataSize)
        local validateData = out.data:sub(numTrainingSamples, dataSize)
        local trainOut = {}
        trainOut.label = trainLabel
        trainOut.data = trainData
        local validateOut = {}
        validateOut.label = validateLabel
        validateOut.data = validateData
        trainOut.data = trainOut.data:double() --Convert to double tensor
        validateOut.data = validateOut.data:double() --Convert to double tensor
        print('Training Data')
        print(trainOut)
        print('Validate Data')
        print(validateOut)
        --Save training and validation tensors
        torch.save(outputTrainFileName, trainOut)
        torch.save(outputValidationFileName, validateOut)
        --Read in test data
        local inputTestFileName = {'cifar-10-batches-bin/test_batch.bin'}
        local outputTestFileName = pathForData .. testFile
        local testOut = convertCifarBinToTorchTensor(inputTestFileName, outputTestFileName)
        testOut.data = testOut.data:double() --Convert to double tensor
        print('Test Data')
        print(testOut)
        torch.save(outputTestFileName, testOut)
    end
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

-- Perform data augmentation on the passed dataset
local function dataAugmentation(passedDataset)
end

--------------------------------------------------------------------------------
--Below are class methods for Cifar10Loader
--------------------------------------------------------------------------------

local Cifar10Loader = torch.class 'Cifar10Loader'

function Cifar10Loader:__init(dataDir, datasetType, sizeRestriction)
    local height = 32
    local width = 32
    local depth = 3

    --Download/convert data if needed and place in passed in dataDir
    prepCifar10Data(dataDir)

    -- Which part of the dataset are we using? (train, validate, test)
    local datasetName 
    if datasetType == 'train' then
        datasetName = dataDir .. '/' .. trainFile
    elseif datasetType == 'validate' then
        datasetName = dataDir .. '/' .. validateFile
    elseif datasetType == 'test' then
        datasetName = dataDir .. '/' .. testFile
    end

    -- Load the data set
    print('Loading ' .. datasetName .. ' data set')
    loadedDataset = torch.load(datasetName)

    -- Add 1 to all labels so that there are no labels with 0
    -- This prevents us from using any loss functions with log likelihood
    -- Set class fields for loaded data set
    for i = 1, loadedDataset.label:size(1) do
        loadedDataset.label[i] = loadedDataset.label[i] + 1
    end

    -- Apply sizeRestriciton, this is used for code validation
    if sizeRestriction then
        loadedDataset.data = loadedDataset.data[{ {1, sizeRestriction}, {}, {}, {} }]
        loadedDataset.label = loadedDataset.label[{ {1, sizeRestriction} }]
        collectgarbage()
        print('Applied Size Restriction: ', sizeRestriction)
    end

    -- Assign class members
    self.dataSize = loadedDataset.data:size(1)
    self.classes = {'airplane', 'automobile', 'bird', 'cat', 
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
    self.index = 1
    self.data = loadedDataset.data
    self.labels = loadedDataset.label

    -- Apply data transformations/normalizations
    normalizeData(self.data)

    -- Sanity Checks
    --[[
    print("Sanity Checks for Object Member Assignments: ")
    print(self.data:size(1))
    print(self.dataSize)
    print(self.classes)
    print(self.index)
    print(type(self.data))
    print(type(self.labels))
    print('--------------End Sanity Check----------------')    
    --]]
end

-- Do not define function if EXCLUDE_CUDA_FLAG is marked
if not EXCLUDE_CUDA_FLAG then
    function Cifar10Loader:cuda()
        self.data = self.data:cuda()
        self.labels = self.labels:cuda()
    end
end

--------------------------------------------------------------------------------
--Test Driver Code
--------------------------------------------------------------------------------
--[[
dataDir = 'torch-data'
sizeRestriction = 100
data_train = Cifar10Loader(dataDir, 'train', sizeRestriction)
data_train = Cifar10Loader(dataDir, 'validate', sizeRestriction)
data_train = Cifar10Loader(dataDir, 'test', sizeRestriction)
--]]