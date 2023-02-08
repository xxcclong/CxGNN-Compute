#include "stitch_kernel.h"

__global__ void pad_rel(Index *rel, Index *idx, const Index *count,
                        const int thres, const int num_rel, const int base) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_rel * (thres - 1)) {
    return;
  }
  int which_rel = tid / (thres - 1);
  int rel_count = count[which_rel];
  int mod = rel_count % thres;
  Index pos = base + tid;
  if ((tid % (thres - 1)) < (thres - mod) && (mod != 0)) {
    rel[pos] = which_rel;
    idx[pos] = -1;
  } else {
    rel[pos] = num_rel;
    idx[pos] = -1;
  }
}

void pad_rel_gpu(Tensor rel, Tensor idx, Tensor count, int thres, int num_rel,
                 Index base) {
  int num_threads = num_rel * (thres - 1);
  int num_blocks = (num_threads + 63) / 64;
  pad_rel<<<num_blocks, 64>>>(rel.data<Index>(), idx.data<Index>(),
                              count.data<Index>(), thres, num_rel, base);
}