#include "spmm_multihead_kernel.h"

__global__ void spmm_multihead_pre_reduce(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ val, const float *__restrict__ vin,
    float *__restrict__ vout, int num_node, int INFEATURE, int num_head) {
  int lane = threadIdx.x & 31;
  int target_id = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int row = target_id;
  //   int head_id = target_id % num_head;
  int col = (threadIdx.y << 5) + lane;  // [0, INFEATURE]
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float rs = 0.0f;
  int theidx;
  float theval;
  int jlimit;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    for (int head_id = 0; head_id < num_head; ++head_id) {
      if (i + lane < end) {
        theidx = idx[i + lane];
        theval = val[(i + lane) * num_head + head_id];
      }
      jlimit = 32;
      if (end - i < 32) jlimit = end - i;
      for (int j = 0; j < jlimit; ++j) {
        int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
        float val = __shfl_sync(0xffffffff, theval, j, 32);
        if (col < INFEATURE)
          rs += vin[(neighbor_id * num_head + head_id) * INFEATURE + col] * val;
      }
    }
  }
  if (col < INFEATURE) vout[target_id * INFEATURE + col] = rs;
}

__global__ void spmm_multihead_sharedmem_4(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ val, const float *__restrict__ vin,
    float *__restrict__ vout, int num_node, int INFEATURE, int num_head,
    int rpb, int cpb, int cpw, int grid_map, int block_map) {
  extern __shared__ int sh[];
  int shared_mem_offset = threadIdx.y * blockDim.x;
  int shared_mem_offset2 = threadIdx.x / 32 * 32;
  int required_num_block = INFEATURE / cpb;
  int a = blockIdx.x / required_num_block * gridDim.y + blockIdx.y;
  int b = blockIdx.x * (gridDim.y / required_num_block) +
          blockIdx.y / required_num_block;
  int block_start_row = grid_map == 0 ? a : b;
  block_start_row *= rpb;
  int block_start_col = grid_map == 0 ? (blockIdx.x % required_num_block) * cpb
                                      : (blockIdx.y % required_num_block) * cpb;
  int required_num_warp = cpb / cpw;
  int c = (threadIdx.x / 32) / required_num_warp * blockDim.y + threadIdx.y;
  int d = (threadIdx.x / 32) * (blockDim.y / required_num_warp) +
          threadIdx.y / required_num_warp;
  int row = block_start_row + (block_map == 0 ? c : d);
  int lane = threadIdx.x & 31;
  int col = block_start_col +
            (block_map == 0 ? ((threadIdx.x / 32) % required_num_warp) * cpw
                            : (threadIdx.y % required_num_warp) * cpw);
  col += lane * cpw / 32;
  Index visit_col = col;
  Index infeat = INFEATURE;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float res0 = 0.f;
  float res1 = 0.f;
  float res2 = 0.f;
  float res3 = 0.f;
  int jlimit;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      sh[shared_mem_offset + threadIdx.x] = (int)(idx[i + lane]);
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = sh[shared_mem_offset + shared_mem_offset2 + j];
      Index target_addr = (Index)neighbor_id * infeat * num_head + visit_col;
      for (int head_id = 0; head_id < num_head; ++head_id) {
        float value = __ldg(val + (i + j) * num_head + head_id);
        res0 += vin[target_addr] * value;
        res1 += vin[target_addr + 1] * value;
        res2 += vin[target_addr + 2] * value;
        res3 += vin[target_addr + 3] * value;
        target_addr += infeat;
      }
    }
  }
  Index target_addr = row * INFEATURE + visit_col;
  vout[target_addr] = res0;
  vout[target_addr + 1] = res1;
  vout[target_addr + 2] = res2;
  vout[target_addr + 3] = res3;
}