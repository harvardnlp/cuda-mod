#include "THCUNN.h"
#include "common.h"

struct MaxFloat
{
  __device__ __forceinline__ float operator()(float max, float v) const
  {
    return fmaxf(max, v);
  }
};

struct SumFloat
{
  __device__ __forceinline__ float operator()(float sum, float v) const
  {
    return sum + v;
  }
};

struct SumExpFloat
{
  __device__ __forceinline__ SumExpFloat(float v)
    : max_k(v)
  {}

  __device__ __forceinline__ float operator()(float sum, float v) const
  {
    return sum + expf(v - max_k);
  }

  const float max_k;
};

struct NoFinal
{
  __device__ __forceinline__ float operator()(float v) const
  {
    return v;
  }
};

struct LSMFinal
{
  __device__ __forceinline__ LSMFinal(float m)
    : max_k(m)
  {}

  __device__ __forceinline__ float operator()(float v) const
  {
    return max_k + logf(v);
  }

  const float max_k;
};

template <typename Reduction, typename Finalize>
__device__ __forceinline__ float
blockReduce(float* smem, float val,
            const Reduction& r,
            float defaultVal,
            const Finalize& f)
{
  // To avoid RaW races from chaining blockReduce calls together, we
  // need a sync here
  __syncthreads();

  smem[threadIdx.x] = val;

  __syncthreads();

  float warpVal = defaultVal;

  // First warp will perform per-warp reductions for the remaining warps
  if ((threadIdx.x / 32) == 0) // only threads in warp1 go into this (if)
  {
    int lane = threadIdx.x % 32; // from 0 to 31

    // if less than 1024 threads per block, then only activate the relevant lanes
    if (lane < blockDim.x / 32)
    {
#pragma unroll
      for (int i = 0; i < 32; ++i)
      {
        warpVal = r(warpVal, smem[lane * 32 + i]);
      }

      smem[lane] = warpVal;
    }
  }

  __syncthreads();

  // First thread will perform a reduction of the above per-warp reductions
  float blockVal = defaultVal;

  if (threadIdx.x == 0)
  {
    for (int i = 0; i < blockDim.x / 32; ++i)
    {
      blockVal = r(blockVal, smem[i]);
    }

    smem[0] = f(blockVal);
  }

  // Sync and broadcast
  __syncthreads();
  return smem[0];
}

template <typename Reduction>
__device__ __forceinline__ float
blockReduce(float* smem, float val,
            const Reduction& r,
            float defaultVal)
{
  return blockReduce<Reduction, NoFinal>(smem, val, r, defaultVal, NoFinal());
}

template <typename Reduction, int ILP>
__device__ __forceinline__ float
ilpReduce(float* data,
          int size,
          const Reduction& r,
          float defaultVal)
{
  float threadVal = defaultVal;
  int offset = threadIdx.x;

  int last = size % (ILP * blockDim.x);

  // Body (unroll by ILP times)
  for (; offset < size - last; offset += blockDim.x * ILP)
  {
    float tmp[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
    {
      tmp[j] = data[offset + j * blockDim.x];
    }

#pragma unroll
    for (int j = 0; j < ILP; ++j)
    {
      threadVal = r(threadVal, tmp[j]);
    }
  }

  // Epilogue
  for (; offset < size; offset += blockDim.x)
  {
    threadVal = r(threadVal, data[offset]);
  }

  return threadVal;
}

template <int ILP>
__global__ void
cunn_LogSum_kernel(float *output, float *input, int elements)
{
  extern __shared__ float buffer[];
  // forward pointers to batch[blockIdx.x]
  // each block handles a sample in the mini-batch
  input += blockIdx.x * elements;

  // find the max of the batch
  float threadMax =
    ilpReduce<MaxFloat, ILP>(input, elements, MaxFloat(), -FLT_MAX);
  // find the max over all batches
  float max_k =
    blockReduce<MaxFloat>(buffer, threadMax, MaxFloat(), -FLT_MAX);

  float threadExp =
    ilpReduce<SumExpFloat, ILP>(input, elements, SumExpFloat(max_k), 0.0f);
  float logsum_k =
    blockReduce<SumFloat, LSMFinal>(
      buffer, threadExp, SumFloat(), 0.0f, LSMFinal(max_k));

    // This is the nan trick.
  if (logsum_k != logsum_k) {
      logsum_k = -CUDART_INF_F;
  }

  output[blockIdx.x] = logsum_k;
}

// Sum over each batch
TH_API void THNN_CudaLogSpace_sum(
    THCState *state, THCudaTensor *output,
    THCudaTensor *input) {

  THCUNN_assertSameGPU(state, 2, input, output);
  input = THCudaTensor_newContiguous(state, input);
  THCudaTensor_resizeAs(state, output, input);

  int batchSize = THCudaTensor_size(state, input, 0);
  int elementSize = THCudaTensor_size(state, input, 1);

  dim3 grid(batchSize);
  dim3 block(1024);

  cunn_LogSum_kernel<2>
    <<<grid, block, block.x * sizeof(float), THCState_getCurrentStream(state)>>>(
      THCudaTensor_data(state, output),
      THCudaTensor_data(state, input),
      elementSize
  );
  THCudaCheck(cudaGetLastError());
  THCudaTensor_free(state, input);
}
