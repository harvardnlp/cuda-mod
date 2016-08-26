#include <THC/THC.h>
#include <THC/THCApply.cuh>

#define THIndexTensor THCudaLongTensor
#define THIndexTensor_(NAME) THCudaLongTensor_ ## NAME

TH_API void THNN_CudaLogSpace_add(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output);
