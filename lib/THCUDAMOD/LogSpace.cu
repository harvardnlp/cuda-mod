#include "THCUNN.h"
#include "common.h"


struct logadd_functor
{
    __device__ void operator()(float* output, const float* input1, const float* input2) const
    {
        if (*input1 < *input2) {
            *output = log(1 + exp(*input1 - *input2)) + *input2;
        } else {
            *output = log(1 + exp(*input2 - *input1)) + *input1;
        }
    }
};

TH_API void THNN_CudaLogSpace_add(
    THCState *state, THCudaTensor *output,
    THCudaTensor *input1, THCudaTensor *input2) {
    THC_pointwiseApply3(state, output, input1, input2, logadd_functor());
}


struct logadd_inplace_functor
{
    __device__ void operator()(float* input1, const float* input2) const
    {
        if (*input1 < *input2) {
            *input1 = log(1 + exp(*input1 - *input2)) + *input2;
        } else {
            *input1 = log(1 + exp(*input2 - *input1)) + *input1;
        }
    }
};


TH_API void THNN_CudaLogSpace_add_inplace(
    THCState *state, THCudaTensor *input1, THCudaTensor *input2) {
    THC_pointwiseApply2(state, input1, input2, logadd_inplace_functor());
}

struct pointwisemod_functor
{
    __device__ void operator()(float* output, const float* input) const
  {
      unsigned int* inta = reinterpret_cast<unsigned int*>(output);
      // Set the last bit to the sign value;
      *inta = *inta ^ ((-((*input) == 1.0) ^ (*inta)) & (1 << 0));
      *output = *(reinterpret_cast<float*>(inta));

  }
};



struct signedAdd_functor
{
    __device__ void operator()(float* output, const float* input1, const float* input2) const
  {
      // Get back the signs
      float t1_sign = (((*reinterpret_cast<const unsigned int*>(input1)) >> 0) & 1) ? 1.0 : -1.0;
      float t2_sign = (((*reinterpret_cast<const unsigned int*>(input2)) >> 0) & 1) ? 1.0 : -1.0;


      // Do the add.
      float mx = max(*input1, *input2);
      float mn = min(*input1, *input2);
      float mn_mx = mn - mx;
      *output = log1p(exp(mn_mx) * t1_sign * t2_sign) + mx;

      // Change sign bit of output.
      float sign = (*input1 > *input2) ? t1_sign : t2_sign;
      unsigned int* inta = reinterpret_cast<unsigned int*>(output);
      *inta = *inta ^ ((-((sign) == 1.0) ^ (*inta)) & (1 << 0));
  }
};

struct getsign_functor
{
    __device__ void operator()(float* output, float* input) const
  {
      *output = (((*reinterpret_cast<const unsigned int*>(input) >> 0) & 1)) ? 1.0 : -1.0;
      // Reset the last bit
      unsigned int* inta = reinterpret_cast<unsigned int*>(input);
      *inta &= ~(1 << 0);

      // This is the nan trick.
      if (*output != *output) {
          *output = -1e10;
      }
  }
};

TH_API void THNN_CudaModSign(
    THCState *state, THCudaTensor *output, THCudaTensor *output_sign)
    {
    THC_pointwiseApply2(state, output, output_sign, pointwisemod_functor());
}

TH_API void THNN_CudaGetSign(
    THCState *state, THCudaTensor *output, THCudaTensor *output_sign)
    {
    THC_pointwiseApply2(state, output_sign, output, getsign_functor());
}


TH_API void THNN_CudaSignedLogSpace_add(
    THCState *state, THCudaTensor *output, THCudaTensor *output_sign,
    THCudaTensor *input1, THCudaTensor *input2,
    THCudaTensor *tensor1_sign, THCudaTensor *tensor2_sign) {

    THC_pointwiseApply2(state, input1, tensor1_sign, pointwisemod_functor());
    THC_pointwiseApply2(state, input2, tensor2_sign, pointwisemod_functor());

    THC_pointwiseApply3(state, output, input1, input2, signedAdd_functor());

    THC_pointwiseApply2(state, output_sign, output, getsign_functor());
}

struct signedAdd_inplace_functor
{
    __device__ void operator()(float* input1, const float* input2, const float* t1t2_sign_prod) const
  {
      float mx = max(*input1, *input2);
      float mn = min(*input1, *input2);
      float mn_mx = mn - mx;
      *input1 = log1p(exp(mn_mx) * *t1t2_sign_prod) + mx;
      if (*input1 != *input1) {
          *input1 = -1e10;
      }
      
  }
};

struct prod_functor
{
    __device__ void operator()(float* output, const float* input1, const float* input2) const
  {
      *output = *input1 * *input2;      
  }
};

struct ge_functor
{
    __device__ void operator()(float* tensor1_sign, const float* ge, const float* tensor2_sign) const
  {
      if (*ge < 1) {
          *tensor1_sign = *tensor2_sign;
      }      
  }
};

struct fixnan_functor
{
    __device__ void operator()(float* output) const
    {
      if (*output != *output) {
          *output = -1 * CUDART_INF;
      }
    }
};

TH_API void THNN_CudaSignedLogSpace_add_inplace(
    THCState *state, THCudaTensor *input1, THCudaTensor *input2,
    THCudaTensor *tensor1_sign, THCudaTensor *tensor2_sign,
    THCudaTensor *t1t2_sign_prod, THCudaTensor *ge) {
    THC_pointwiseApply3(state, t1t2_sign_prod, tensor1_sign, tensor2_sign, prod_functor());
    THC_pointwiseApply3(state, tensor1_sign, ge, tensor2_sign, ge_functor());
    THC_pointwiseApply3(state, input1, input2, t1t2_sign_prod, signedAdd_inplace_functor());
}




TH_API void THNN_CudaFixNaN(
    THCState *state, THCudaTensor *input) {
    THC_pointwiseApply1(state, input, fixnan_functor());
}



// void THNN_CudaLogSpace_abs(THCState *state, THCudaTensor *output,
//                            THCudaTensor *input)
// {
//   THC_pointwiseApply2(state, output, input, abs_functor());
// }
// struct addexpOutput_functor
// {
//     const float max_;

//     addexpOutput_functor(float max)
//     : max_(max)
//   {}


//   __device__ void operator()(float* output, const float* input) const
//   {
//     *output = exp(*input - max_);
//   }
// };

// struct addexpOutputSign_functor
// {
//     const float max_;

//     addexpOutputSign_functor(float max)
//     : max_(max)
//   {}


//     __device__ void operator()(float* output, const float* input, const float* input_sign) const
//   {
//       *output = exp(*input - max_) * (*input_sign);
//   }
// };


// struct logaddOutput_functor
// {
//     const float max_;

//     logaddOutput_functor(float max)
//     : max_(max)
//   {}


//   __device__ void operator()(float* output, const float* input) const
//   {
//       *output = log(fabs(*input)) + max_;
//       if (*output != *output)
//           *output = -1e10;
//   }
// };

// void THNN_CudaLogSpace_bmm(THCState *state, THCudaTensor *output,
//                            THCudaTensor *input1, THCudaTensor *input2,
//                            THCudaTensor *tmp1, THCudaTensor *tmp2)
// {
//   THCUNN_assertSameGPU(state, 2, input1, output);
//   THCUNN_assertSameGPU(state, 2, input2, output);
//   THCUNN_assertSameGPU(state, 2, tmp1, output);
//   THCUNN_assertSameGPU(state, 2, tmp2, output);
//   // THCudaTensor_resizeAs(state, output, input1);
//   //find maxes
//   float max1 = THCudaTensor_maxall(state, input1);
//   float max2 = THCudaTensor_maxall(state, input2);

//   THC_pointwiseApply2(state, tmp1, input1, addexpOutput_functor(max1));
//   THC_pointwiseApply2(state, tmp2, input2, addexpOutput_functor(max2));

//   // call bmm
//   THCudaTensor_baddbmm(state, output, 0.0, output, 1.0, tmp1, tmp2);
//   THC_pointwiseApply2(state, output, output, logaddOutput_functor(max1 + max2));
// }


// void THNN_CudaSignedLogSpace_bmm(THCState *state, THCudaTensor *output, THCudaTensor *output_sign,
//                                  THCudaTensor *input1, THCudaTensor *input2,
//                                  THCudaTensor *tensor1_sign, THCudaTensor *tensor2_sign,
//                                  THCudaTensor *tmp1, THCudaTensor *tmp2)
// {
//   float max1 = THCudaTensor_maxall(state, input1);
//   float max2 = THCudaTensor_maxall(state, input2);

//   THC_pointwiseApply3(state, tmp1, input1, tensor1_sign, addexpOutputSign_functor(max1));
//   THC_pointwiseApply3(state, tmp2, input2, tensor2_sign, addexpOutputSign_functor(max2));


//   // call bmm
//   THCudaTensor_baddbmm(state, output, 0.0, output, 1.0, tmp1, tmp2);
//   THCudaTensor_sign(state, output_sign, output);
//   THC_pointwiseApply2(state, output, output, logaddOutput_functor(max1 + max2));
// }




// struct signedAdd_functor
// {
//     __device__ void operator()(float* output, const float* input1, const float* input2) const
//   {
//       float mx = max(*input1, *input2);
//       float mn = min(*input1, *input2);
//       float t1_sign = (((*reinterpret_cast<const unsigned int*>(input1)) >> 0) & 1) ? 1.0 : -1.0;
//       float t2_sign = (((*reinterpret_cast<const unsigned int*>(input2)) >> 0) & 1) ? 1.0 : -1.0;
//       float mn_mx = mn - mx;
//       *output = log1p(exp(mn_mx) * t1_sign * t2_sign) + mx;
//       float sign = (*input1 > *input2) ? t1_sign : t2_sign;

//       // Change sign bit of output.
//       unsigned int* inta = reinterpret_cast<unsigned int*>(output);
//       *inta = *inta ^ ((-((sign) == 1.0) ^ (*inta)) & (1 << 0));
//       // *output = *(reinterpret_cast<float*>(inta));
//   }
// };

// struct getsign_functor
// {
//     __device__ void operator()(float* output, float* input) const
//   {
//       *output = (((*reinterpret_cast<const unsigned int*>(input) >> 0) & 1)) ? 1.0 : -1.0;
//       // Reset the last bit
//       unsigned int* inta = reinterpret_cast<unsigned int*>(input);
//       *inta &= ~(1 << 0);
//       if (*output != *output) {
//           *output = -1e10;
//       }
//   }
// };



// TH_API void THNN_CudaSignedLogSpace_add(
//     THCState *state, THCudaTensor *output, THCudaTensor *output_sign,
//     THCudaTensor *input1, THCudaTensor *input2,
//     THCudaTensor *tensor1_sign, THCudaTensor *tensor2_sign) {

//     THC_pointwiseApply2(state, input1, tensor1_sign, pointwisemod_functor());
//     THC_pointwiseApply2(state, input2, tensor2_sign, pointwisemod_functor());

//     THC_pointwiseApply3(state, output, input1, input2, signedAdd_functor());

//     THC_pointwiseApply2(state, output_sign, output, getsign_functor());
// }


// TH_API void THNN_CudaSignedLogSpace_addSimple(
//     THCState *state, THCudaTensor *output,
//     THCudaTensor *input1, THCudaTensor *input2) {
//     THC_pointwiseApply3(state, output, input1, input2, signedAdd_functor());
// }




// TH_API void THNN_CudaSignedLogSpace_sum(
//     THCState *state, THCudaTensor *input, int dim) {
//     THC_pointwiseApply3(state, output, input1, input2, signedAdd_functor());
// }


// struct sum_functor
// {
//     const float max_;

//     logaddOutput_functor(float max)
//     : max_(max)
//     {}


//     __device__ void operator()(float* output, float* input) const
//   {
//       float sign = (((*reinterpret_cast<const unsigned int*>(input)) >> 0) & 1) ? 1.0 : -1.0;
//       *output = exp(tensor - max_) * sign;
//   }
// };


// TH_API void THNN_CudaSignedLogSpace_sumNumber(
//     THCState *state, THCudaTensor *input) {
//     float max1 = THCudaTensor_maxall(state, input1);

//     THC_pointwiseApply3(state, output, input1, input2, signedAdd_functor());
//     // float sum =
// }
