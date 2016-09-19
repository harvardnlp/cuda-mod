#include <THC/THC.h>
#include <THC/THCApply.cuh>

#define THIndexTensor THCudaLongTensor
#define THIndexTensor_(NAME) THCudaLongTensor_ ## NAME

#define CUDART_INF_F __int_as_float(0x7f800000)

TH_API void THNN_CudaLogSpace_add(
    THCState *state, THCudaTensor *output,
    THCudaTensor *input1, THCudaTensor *input2);

TH_API void THNN_CudaLogSpace_add_inplace(
    THCState *state, THCudaTensor *input1, THCudaTensor *input2);

TH_API void THNN_CudaModSign(
    THCState *state, THCudaTensor *input, THCudaTensor *input_sign);

TH_API void THNN_CudaGetSign(
    THCState *state, THCudaTensor *input, THCudaTensor *input_sign);

TH_API void THNN_CudaSignedLogSpace_add(
    THCState *state, THCudaTensor *output, THCudaTensor *output_sign,
    THCudaTensor *input1, THCudaTensor *input2,
    THCudaTensor *tensor1_sign, THCudaTensor *tensor2_sign);

TH_API void THNN_CudaSignedLogSpace_add_inplace(
    THCState *state, THCudaTensor *input1, THCudaTensor *input2,
    THCudaTensor *tensor1_sign, THCudaTensor *tensor2_sign,
    THCudaTensor *t1t2_prod_sign, THCudaTensor *ge);

TH_API void THNN_CudaLogSpace_sum(
    THCState *state, THCudaTensor *output,
    THCudaTensor *input);

TH_API void THNN_CudaFixNaN(
    THCState *state, THCudaTensor *input);
