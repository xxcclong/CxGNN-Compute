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

__global__ void sddmm_multihead_vertex_centric_1(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ src_feat, const float *__restrict__ dst_feat,
    float *__restrict__ output, int num_node, int INFEATURE, int num_head,
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
  // __initialize__
  float item0 = dst_feat[row * INFEATURE + visit_col];
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
      for (int head_id = 0; head_id < num_head; ++head_id) {
        // __computation__
        Index target_addr =
            (Index)(neighbor_id * num_head + head_id) * infeat + visit_col;
        float res = src_feat[target_addr] * item0;
        for (int k = 16; k > 0; k >>= 1) {
          res += __shfl_down_sync(0xffffffff, res, k);  // sum
        }
        if (lane == 0) {
          atomicAdd(output + (i + j) * num_head + head_id, res);
        }
      }
    }
  }
}

__global__ void sddmm_multihead_vertex_centric_2(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ src_feat, const float *__restrict__ dst_feat,
    float *__restrict__ output, int num_node, int INFEATURE, int num_head,
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
  // __initialize__
  float item0 = dst_feat[row * INFEATURE + visit_col];
  float item1 = dst_feat[row * INFEATURE + visit_col + 1];
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
      for (int head_id = 0; head_id < num_head; ++head_id) {
        // __computation__
        Index target_addr =
            (Index)(neighbor_id * num_head + head_id) * infeat + visit_col;
        float res =
            src_feat[target_addr] * item0 + src_feat[target_addr + 1] * item1;
        for (int k = 16; k > 0; k >>= 1) {
          res += __shfl_down_sync(0xffffffff, res, k);  // sum
        }
        if (lane == 0) {
          atomicAdd(output + (i + j) * num_head + head_id, res);
        }
      }
    }
  }
}

__global__ void sddmm_multihead_vertex_centric_4(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ src_feat, const float *__restrict__ dst_feat,
    float *__restrict__ output, int num_node, int INFEATURE, int num_head,
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
  // __initialize__
  float item0 = dst_feat[row * INFEATURE + visit_col];
  float item1 = dst_feat[row * INFEATURE + visit_col + 1];
  float item2 = dst_feat[row * INFEATURE + visit_col + 2];
  float item3 = dst_feat[row * INFEATURE + visit_col + 3];
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
      for (int head_id = 0; head_id < num_head; ++head_id) {
        // __computation__
        Index target_addr =
            (Index)(neighbor_id * num_head + head_id) * infeat + visit_col;
        float res = src_feat[target_addr] * item0 +
                    src_feat[target_addr + 1] * item1 +
                    src_feat[target_addr + 2] * item2 +
                    src_feat[target_addr + 3] * item3;
        for (int k = 16; k > 0; k >>= 1) {
          res += __shfl_down_sync(0xffffffff, res, k);  // sum
        }
        if (lane == 0) {
          atomicAdd(output + (i + j) * num_head + head_id, res);
        }
      }
    }
  }
}

__global__ void sddmm_multihead_vertex_centric_8(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ src_feat, const float *__restrict__ dst_feat,
    float *__restrict__ output, int num_node, int INFEATURE, int num_head,
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
  // __initialize__
  float item0 = dst_feat[row * INFEATURE + visit_col];
  float item1 = dst_feat[row * INFEATURE + visit_col + 1];
  float item2 = dst_feat[row * INFEATURE + visit_col + 2];
  float item3 = dst_feat[row * INFEATURE + visit_col + 3];
  float item4 = dst_feat[row * INFEATURE + visit_col + 4];
  float item5 = dst_feat[row * INFEATURE + visit_col + 5];
  float item6 = dst_feat[row * INFEATURE + visit_col + 6];
  float item7 = dst_feat[row * INFEATURE + visit_col + 7];
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
      for (int head_id = 0; head_id < num_head; ++head_id) {
        // __computation__
        Index target_addr =
            (Index)(neighbor_id * num_head + head_id) * infeat + visit_col;
        float res = src_feat[target_addr] * item0 +
                    src_feat[target_addr + 1] * item1 +
                    src_feat[target_addr + 2] * item2 +
                    src_feat[target_addr + 3] * item3 +
                    src_feat[target_addr + 4] * item4 +
                    src_feat[target_addr + 5] * item5 +
                    src_feat[target_addr + 6] * item6 +
                    src_feat[target_addr + 7] * item7;
        for (int k = 16; k > 0; k >>= 1) {
          res += __shfl_down_sync(0xffffffff, res, k);  // sum
        }
        if (lane == 0) {
          atomicAdd(output + (i + j) * num_head + head_id, res);
        }
      }
    }
  }
}