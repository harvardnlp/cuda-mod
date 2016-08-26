#include <THC/THC.h>
#include <THC/THCApply.cuh>

#define THIndexTensor THCudaLongTensor
#define THIndexTensor_(NAME) THCudaLongTensor_ ## NAME

TH_API void THNN_CudaLogSpace_add(
    THCState *state, THCudaTensor *output,
    THCudaTensor *input1, THCudaTensor *input2);

TH_API void THNN_CudaSignedLogSpace_add(
    THCState *state, THCudaTensor *output, THCudaTensor *output_sign,
    THCudaTensor *input1, THCudaTensor *input2,
    THCudaTensor *tensor1_sign, THCudaTensor *tensor2_sign);
