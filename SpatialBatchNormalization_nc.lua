SpatialBatchNormalization_nc, _ = torch.class('nn.SpatialBatchNormalization_nc', 'nn.SpatialBatchNormalization')

function SpatialBatchNormalization_nc:updateGradInput(input, gradOutput)
   assert(input:dim() == 4, 'only mini-batch supported')
   assert(gradOutput:dim() == 4, 'only mini-batch supported')
   assert(self.train == true, 'should be in training mode when self.train is true')
   local nBatch = input:size(1)
   local nFeature = input:size(2)
   local iH = input:size(3)
   local iW = input:size(4)

   self.gradInput:cmul(self.centered, gradOutput)
   local gi_folded = self.gradInput:view(nBatch, nFeature, iH * iW)
   self.buffer2:mean(self.buffer:mean(gi_folded, 1), 3)
   self.gradInput:repeatTensor(self.buffer2:view(1, nFeature, 1, 1),
                               nBatch, 1, iH, iW)
   self.gradInput:cmul(self.centered):mul(-1)
   self.buffer:repeatTensor(self.std:view(1, nFeature, 1, 1),
                            nBatch, 1, iH, iW)
   self.gradInput:cmul(self.buffer):cmul(self.buffer)

    -- modified code:
    if gradOutput:isContiguous() then
        -- gradOutput = gradOutput:view(size) -- doesn't work with non-contiguous tensors
        self.buffer:mean(gradOutput:view(nBatch, nFeature, iH * iW), 1)
    else
        -- gradOutput = gradOutput:resize(size) -- slower because of memory reallocation and changes gradOutput
        -- gradOutput = gradOutput:clone():resize(size) -- doesn't change gradOutput; safer and even slower
        self.buffer:mean(gradOutput:resize(nBatch, nFeature, iH * iW), 1)
    end

   self.buffer2:mean(self.buffer, 3)
   self.buffer:repeatTensor(self.buffer2:view(1, nFeature, 1, 1),
                            nBatch, 1, iH, iW)
   self.gradInput:add(gradOutput):add(-1, self.buffer)
   self.buffer:repeatTensor(self.std:view(1, nFeature, 1, 1),
                            nBatch, 1, iH, iW)
   self.gradInput:cmul(self.buffer)

   if self.affine then
      self.buffer:repeatTensor(self.weight:view(1, nFeature, 1, 1),
                               nBatch, 1, iH, iW)
      self.gradInput:cmul(self.buffer)
   end

   return self.gradInput
end