#include "sddmm_kernel.h"

__global__ void sddmm_multihead(const Index *__restrict__ src,
                                const Index *__restrict__ dst,
                                const float *__restrict__ src_feat,
                                const float *__restrict__ dst_feat,
                                float *__restrict__ output, Index num_edge,
                                int INFEATURE, int num_head) {
  int lane = threadIdx.x % 32;
  int target_id = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  if (target_id > num_edge * num_head) return;
  int head_id = target_id % num_head;
  int edge_id = target_id / num_head;
  float res = 0.f;
  for (int i = lane; i < INFEATURE; i += 32) {
    res += src_feat[(src[edge_id] * num_head + head_id) * INFEATURE + i] *
           dst_feat[dst[edge_id] * INFEATURE + i];
  }
  for (int k = 16; k > 0; k >>= 1) {
    res += __shfl_down_sync(0xffffffff, res, k);  // sum
  }
  if (lane == 0) {
    output[target_id] = res;
  }
}