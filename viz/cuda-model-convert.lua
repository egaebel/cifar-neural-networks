require 'torch'
require 'nn'
require 'cunn'
--require 'paths'

local function convertModels(specificName)
    local createdFiles = {}
    for f in paths.files('../model-nets') do
        if specificName == nil or f == specificName then
            local floatIndicatorIndex = string.find(f, '--float')
            if f ~= '.' and f ~= '..' and not floatIndicatorIndex then
                local modelPath = paths.concat('../model-nets', f)
                print("Reading Model:")
                print(modelPath)
                local model = torch.load(modelPath)
                model = model:float()
                print(model)
                local extensionIndex = string.find(modelPath, '%.net')
                modelPath = string.sub(modelPath, 1, (extensionIndex - 1))
                --modelPath = modelPath .. '--float.net'
                modelPath = 'inception' .. '--float.net'
                print("Saving float model to: ")
                print(modelPath)
                torch.save(modelPath, model)
                table.insert(createdFiles, modelPath)
            end
        end
    end
    print("Created files:")
    print(createdFiles)
end

convertModels('model.net')