#include "THCUNN.h"
#include "common.h"

struct absupdateOutput_functor
{
  __device__ void operator()(float* output, const float* input) const
  {
    *output = abs(*input);
  }
};

void THNN_CudaLogSpace_add(THCState *state, THCudaTensor *input, THCudaTensor *output)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCudaTensor_resizeAs(state, output, input);
  THC_pointwiseApply2(state, output, input, absupdateOutput_functor());
}
