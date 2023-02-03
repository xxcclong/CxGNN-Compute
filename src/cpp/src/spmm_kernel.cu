#include "spmm_kernel.h"

__global__ void run_spmm(const Index *__restrict__ ptr,
                         const Index *__restrict__ idx,
                         const float *__restrict__ vin,
                         float *__restrict__ vout, int num_node, int INFEATURE,
                         int rpb, int cpb, int cpw, int grid_map,
                         int block_map) {
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
  float res = 0.f;
  int theidx;
  int jlimit;
#pragma unroll
  for (int offset = 0; offset < cpw; offset += 32) {
    visit_col = col + offset;
    for (Index i = begin; i < end; i += 32) {
      if (i + lane < end) {
        theidx = idx[i + lane];
      }
      jlimit = 32;
      if (end - i < 32) jlimit = end - i;
      for (int j = 0; j < jlimit; ++j) {
        int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
        Index target_addr = (Index)neighbor_id * infeat + visit_col;
        res += vin[target_addr];
      }
    }
    Index target_addr = row * INFEATURE + visit_col;
    vout[target_addr] = res;
  }
}

__global__ void run_spmm_2(const Index *__restrict__ ptr,
                           const Index *__restrict__ idx,
                           const float *__restrict__ vin,
                           float *__restrict__ vout, int num_node,
                           int INFEATURE, int rpb, int cpb, int cpw,
                           int grid_map, int block_map) {
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
  float res1 = 0.f;
  float res2 = 0.f;
  int theidx;
  int jlimit;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      Index target_addr = (Index)neighbor_id * infeat + visit_col;
      res1 += vin[target_addr];
      res2 += vin[target_addr + 1];
    }
  }
  Index target_addr = row * INFEATURE + visit_col;
  vout[target_addr] = res1;
  vout[target_addr + 1] = res2;
}

__global__ void run_spmm_4(const Index *__restrict__ ptr,
                           const Index *__restrict__ idx,
                           const float *__restrict__ vin,
                           float *__restrict__ vout, int num_node,
                           int INFEATURE, int rpb, int cpb, int cpw,
                           int grid_map, int block_map) {
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
  float res1 = 0.f;
  float res2 = 0.f;
  float res3 = 0.f;
  float res4 = 0.f;
  int theidx;
  int jlimit;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      Index target_addr = (Index)neighbor_id * infeat + visit_col;
      res1 += vin[target_addr];
      res2 += vin[target_addr + 1];
      res3 += vin[target_addr + 2];
      res4 += vin[target_addr + 3];
    }
  }
  Index target_addr = row * INFEATURE + visit_col;
  vout[target_addr] = res1;
  vout[target_addr + 1] = res2;
  vout[target_addr + 2] = res3;
  vout[target_addr + 3] = res4;
}

__global__ void run_spmm_8(const Index *__restrict__ ptr,
                           const Index *__restrict__ idx,
                           const float *__restrict__ vin,
                           float *__restrict__ vout, int num_node,
                           int INFEATURE, int rpb, int cpb, int cpw,
                           int grid_map, int block_map) {
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
  float res1 = 0.f;
  float res2 = 0.f;
  float res3 = 0.f;
  float res4 = 0.f;
  float res5 = 0.f;
  float res6 = 0.f;
  float res7 = 0.f;
  float res8 = 0.f;
  int theidx;
  int jlimit;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      Index target_addr = (Index)neighbor_id * infeat + visit_col;
      res1 += vin[target_addr];
      res2 += vin[target_addr + 1];
      res3 += vin[target_addr + 2];
      res4 += vin[target_addr + 3];
      res5 += vin[target_addr + 4];
      res6 += vin[target_addr + 5];
      res7 += vin[target_addr + 6];
      res8 += vin[target_addr + 7];
    }
  }
  Index target_addr = row * INFEATURE + visit_col;
  vout[target_addr] = res1;
  vout[target_addr + 1] = res2;
  vout[target_addr + 2] = res3;
  vout[target_addr + 3] = res4;
  vout[target_addr + 4] = res5;
  vout[target_addr + 5] = res6;
  vout[target_addr + 6] = res7;
  vout[target_addr + 7] = res8;
}

__global__ void run_spmm_sharedmem(const Index *__restrict__ ptr,
                                   const Index *__restrict__ idx,
                                   const float *__restrict__ vin,
                                   float *__restrict__ vout, int num_node,
                                   int INFEATURE, int rpb, int cpb, int cpw,
                                   int grid_map, int block_map) {
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
  float res = 0.f;
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
      Index target_addr = (Index)neighbor_id * infeat + visit_col;
      res += vin[target_addr];
    }
  }
  Index target_addr = row * INFEATURE + visit_col;
  vout[target_addr] = res;
}

__global__ void run_spmm_sharedmem_int(const int *__restrict__ ptr,
                                       const int *__restrict__ idx,
                                       const float *__restrict__ vin,
                                       float *__restrict__ vout, int num_node,
                                       int INFEATURE, int rpb, int cpb, int cpw,
                                       int grid_map, int block_map) {
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
  int visit_col = col;
  int infeat = INFEATURE;
  if (row >= num_node) return;
  int begin = ptr[row], end = ptr[row + 1];
  float res = 0.f;
  int jlimit;
#pragma unroll
  for (int i = begin; i < end; i += 32) {
    if (i + lane < end) {
      sh[shared_mem_offset + threadIdx.x] = (int)(idx[i + lane]);
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = sh[shared_mem_offset + shared_mem_offset2 + j];
      int target_addr = neighbor_id * infeat + visit_col;
      res += vin[target_addr];
    }
  }
  int target_addr = row * INFEATURE + visit_col;
  vout[target_addr] = res;
}

__global__ void run_spmm_sharedmem_2(const Index *__restrict__ ptr,
                                     const Index *__restrict__ idx,
                                     const float *__restrict__ vin,
                                     float *__restrict__ vout, int num_node,
                                     int INFEATURE, int rpb, int cpb, int cpw,
                                     int grid_map, int block_map) {
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
      Index target_addr = (Index)neighbor_id * infeat + visit_col;
      res0 += vin[target_addr];
      res1 += vin[target_addr + 1];
    }
  }
  Index target_addr = row * INFEATURE + visit_col;
  vout[target_addr] = res0;
  vout[target_addr + 1] = res1;
}

__global__ void run_spmm_sharedmem_4(const Index *__restrict__ ptr,
                                     const Index *__restrict__ idx,
                                     const float *__restrict__ vin,
                                     float *__restrict__ vout, int num_node,
                                     int INFEATURE, int rpb, int cpb, int cpw,
                                     int grid_map, int block_map) {
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
      Index target_addr = (Index)neighbor_id * infeat + visit_col;
      res0 += vin[target_addr];
      res1 += vin[target_addr + 1];
      res2 += vin[target_addr + 2];
      res3 += vin[target_addr + 3];
    }
  }
  Index target_addr = row * INFEATURE + visit_col;
  vout[target_addr] = res0;
  vout[target_addr + 1] = res1;
  vout[target_addr + 2] = res2;
  vout[target_addr + 3] = res3;
}

__global__ void run_spmm_sharedmem_8(const Index *__restrict__ ptr,
                                     const Index *__restrict__ idx,
                                     const float *__restrict__ vin,
                                     float *__restrict__ vout, int num_node,
                                     int INFEATURE, int rpb, int cpb, int cpw,
                                     int grid_map, int block_map) {
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
  float res4 = 0.f;
  float res5 = 0.f;
  float res6 = 0.f;
  float res7 = 0.f;
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
      Index target_addr = (Index)neighbor_id * infeat + visit_col;
      res0 += vin[target_addr];
      res1 += vin[target_addr + 1];
      res2 += vin[target_addr + 2];
      res3 += vin[target_addr + 3];
      res4 += vin[target_addr + 4];
      res5 += vin[target_addr + 5];
      res6 += vin[target_addr + 6];
      res7 += vin[target_addr + 7];
    }
  }
  Index target_addr = row * INFEATURE + visit_col;
  vout[target_addr] = res0;
  vout[target_addr + 1] = res1;
  vout[target_addr + 2] = res2;
  vout[target_addr + 3] = res3;
  vout[target_addr + 4] = res4;
  vout[target_addr + 5] = res5;
  vout[target_addr + 6] = res6;
  vout[target_addr + 7] = res7;
}

__global__ void run_spmm_sharedmem_step_2(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ vin, float *__restrict__ vout, int num_node,
    int INFEATURE, int rpb, int cpb, int cpw, int grid_map, int block_map) {
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
  col += lane;
  Index visit_col = col;
  Index infeat = INFEATURE;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float res0 = 0.f;
  float res1 = 0.f;
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
      Index target_addr = (Index)neighbor_id * infeat + visit_col;
      res0 += vin[target_addr];
      res1 += vin[target_addr + 32];
    }
  }
  Index target_addr = row * INFEATURE + visit_col;
  vout[target_addr] = res0;
  vout[target_addr + 32] = res1;
}

__global__ void run_spmm_sharedmem_step_4(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ vin, float *__restrict__ vout, int num_node,
    int INFEATURE, int rpb, int cpb, int cpw, int grid_map, int block_map) {
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
  col += lane;  // no scale in step implementation
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
      Index target_addr = (Index)neighbor_id * infeat + visit_col;
      res0 += vin[target_addr];
      res1 += vin[target_addr + 32];
      res2 += vin[target_addr + 64];
      res3 += vin[target_addr + 96];
    }
  }
  Index target_addr = row * INFEATURE + visit_col;
  vout[target_addr] = res0;
  vout[target_addr + 32] = res1;
  vout[target_addr + 64] = res2;
  vout[target_addr + 96] = res3;
}

__global__ void run_spmm_sharedmem_step_2_int(
    const int *__restrict__ ptr, const int *__restrict__ idx,
    const float *__restrict__ vin, float *__restrict__ vout, int num_node,
    int INFEATURE, int rpb, int cpb, int cpw, int grid_map, int block_map) {
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
  col += lane;
  int visit_col = col;
  int infeat = INFEATURE;
  if (row >= num_node) return;
  int begin = ptr[row], end = ptr[row + 1];
  float res0 = 0.f;
  float res1 = 0.f;
  int jlimit;
#pragma unroll
  for (int i = begin; i < end; i += 32) {
    if (i + lane < end) {
      sh[shared_mem_offset + threadIdx.x] = (int)(idx[i + lane]);
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = sh[shared_mem_offset + shared_mem_offset2 + j];
      int target_addr = neighbor_id * infeat + visit_col;
      res0 += vin[target_addr];
      res1 += vin[target_addr + 32];
    }
  }
  int target_addr = row * INFEATURE + visit_col;
  vout[target_addr] = res0;
  vout[target_addr + 32] = res1;
}

__global__ void run_spmm_sharedmem_step_4_int(
    const int *__restrict__ ptr, const int *__restrict__ idx,
    const float *__restrict__ vin, float *__restrict__ vout, int num_node,
    int INFEATURE, int rpb, int cpb, int cpw, int grid_map, int block_map) {
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
  col += lane;  // no scale in step implementation
  int visit_col = col;
  int infeat = INFEATURE;
  if (row >= num_node) return;
  int begin = ptr[row], end = ptr[row + 1];
  float res0 = 0.f;
  float res1 = 0.f;
  float res2 = 0.f;
  float res3 = 0.f;
  int jlimit;
#pragma unroll
  for (int i = begin; i < end; i += 32) {
    if (i + lane < end) {
      sh[shared_mem_offset + threadIdx.x] = (int)(idx[i + lane]);
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = sh[shared_mem_offset + shared_mem_offset2 + j];
      int target_addr = neighbor_id * infeat + visit_col;
      res0 += vin[target_addr];
      res1 += vin[target_addr + 32];
      res2 += vin[target_addr + 64];
      res3 += vin[target_addr + 96];
    }
  }
  int target_addr = row * INFEATURE + visit_col;
  vout[target_addr] = res0;
  vout[target_addr + 32] = res1;
  vout[target_addr + 64] = res2;
  vout[target_addr + 96] = res3;
}

__global__ void run_spmm_sharedmem_step_8_int(
    const int *__restrict__ ptr, const int *__restrict__ idx,
    const float *__restrict__ vin, float *__restrict__ vout, int num_node,
    int INFEATURE, int rpb, int cpb, int cpw, int grid_map, int block_map) {
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
  col += lane;  // no scale in step implementation
  int visit_col = col;
  int infeat = INFEATURE;
  if (row >= num_node) return;
  int begin = ptr[row], end = ptr[row + 1];
  float res0 = 0.f;
  float res1 = 0.f;
  float res2 = 0.f;
  float res3 = 0.f;
  float res4 = 0.f;
  float res5 = 0.f;
  float res6 = 0.f;
  float res7 = 0.f;
  int jlimit;
#pragma unroll
  for (int i = begin; i < end; i += 32) {
    if (i + lane < end) {
      sh[shared_mem_offset + threadIdx.x] = (int)(idx[i + lane]);
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = sh[shared_mem_offset + shared_mem_offset2 + j];
      int target_addr = neighbor_id * infeat + visit_col;
      res0 += vin[target_addr];
      res1 += vin[target_addr + 32];
      res2 += vin[target_addr + 64];
      res3 += vin[target_addr + 96];
      res4 += vin[target_addr + 128];
      res5 += vin[target_addr + 160];
      res6 += vin[target_addr + 192];
      res7 += vin[target_addr + 224];
    }
  }
  int target_addr = row * INFEATURE + visit_col;
  vout[target_addr] = res0;
  vout[target_addr + 32] = res1;
  vout[target_addr + 64] = res2;
  vout[target_addr + 96] = res3;
  vout[target_addr + 128] = res4;
  vout[target_addr + 160] = res5;
  vout[target_addr + 192] = res6;
  vout[target_addr + 224] = res7;
}