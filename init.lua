require "cutorch"
require "nn"
require "cuda-mod.THCUNN"

nn.Module._flattenTensorBuffer['torch.CudaTensor'] = torch.FloatTensor.new
