#include "aggr_kernel.h"

__global__ void fwd_sum_all_x(Index *ptr, Index *idx, float *vin, float *vout,
                              int num_node, int INFEATURE) {
  int col = (threadIdx.x * 4) % INFEATURE;
  Index visit_col = col;
  Index infeat = INFEATURE;
  int lane = threadIdx.x & 31;
  // int row = blockIdx.x * blockDim.y + threadIdx.y;
  int row = (blockDim.x * 4 / INFEATURE * blockIdx.x) +
            ((threadIdx.x * 4) / INFEATURE);
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float4 rs;
  rs.x = 0.f;
  rs.y = 0.f;
  rs.z = 0.f;
  rs.w = 0.f;
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
      // if (col < INFEATURE) {
      Index target_addr = (Index)neighbor_id * infeat + visit_col;
      rs.x += vin[target_addr];
      rs.y += vin[target_addr + 1];
      rs.z += vin[target_addr + 2];
      rs.w += vin[target_addr + 3];
      // }
    }
  }
  // if (col < INFEATURE) {
  Index target_addr = row * INFEATURE + col;
  vout[target_addr] = rs.x;
  vout[target_addr + 1] = rs.y;
  vout[target_addr + 2] = rs.z;
  vout[target_addr + 3] = rs.w;
  // }
}

__global__ void gen_fwd_mean(Index *ptr, Index *idx, float *vin, float *vout,
                             int num_node, int INFEATURE) {
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  int num_neighbor = end - begin;
  float rs = 0.0f;
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
      if (col < INFEATURE) rs += vin[neighbor_id * INFEATURE + col];
    }
  }
  if (col < INFEATURE && num_neighbor != 0)
    vout[row * INFEATURE + col] = (rs) / (num_neighbor);
}

__global__ void gen_fwd_sum(Index *ptr, Index *idx, float *vin, float *vout,
                            int num_node, int INFEATURE) {
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float rs = 0.0f;
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
      if (col < INFEATURE) rs += vin[((Index)neighbor_id) * INFEATURE + col];
    }
  }
  if (col < INFEATURE) vout[row * INFEATURE + col] = rs;
}

__global__ void gen_fwd_mean_edge_value(Index *ptr, Index *idx,
                                        float *edge_value, float *vin,
                                        float *vout, int num_node,
                                        int INFEATURE) {
  assert(false);
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  int num_neighbor = end - begin;
  float rs = 0.0f;
  int theidx;
  float theval;
  int jlimit;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
      theval = edge_value[i + lane];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      float val = __shfl_sync(0xffffffff, theval, j, 32);
      if (col < INFEATURE) rs += vin[neighbor_id * INFEATURE + col] * val;
    }
  }
  if (col < INFEATURE && num_neighbor != 0)
    vout[row * INFEATURE + col] = (rs) / (num_neighbor);
}

__global__ void gen_fwd_sum_edge_value(Index *ptr, Index *idx,
                                       float *edge_value, float *vin,
                                       float *vout, int num_node,
                                       int INFEATURE) {
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float rs = 0.0f;
  int theidx;
  float theval;
  int jlimit;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
      theval = edge_value[i + lane];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      float val = __shfl_sync(0xffffffff, theval, j, 32);
      if (col < INFEATURE) rs += vin[neighbor_id * INFEATURE + col] * val;
    }
  }
  if (col < INFEATURE) vout[row * INFEATURE + col] = rs;
}

__global__ void gen_fwd_mean_edge_value_multi_head(Index *ptr, Index *idx,
                                                   float *edge_value,
                                                   float *vin, float *vout,
                                                   int num_node, int INFEATURE,
                                                   int num_head) {
  assert(false);
  int lane = threadIdx.x & 31;
  int target_id = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int row = target_id / num_head;
  int head_id = target_id % num_head;
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  int num_neighbor = end - begin;
  float rs = 0.0f;
  int theidx;
  float theval;
  int jlimit;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
      theval = edge_value[(i + lane) * num_head + head_id];
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
  if (col < INFEATURE && num_neighbor != 0)
    vout[target_id * INFEATURE + col] = (rs) / (num_neighbor);
}

__global__ void gen_fwd_sum_edge_value_multi_head(Index *ptr, Index *idx,
                                                  float *edge_value, float *vin,
                                                  float *vout, int num_node,
                                                  int INFEATURE, int num_head) {
  int lane = threadIdx.x & 31;
  int target_id = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int row = target_id / num_head;
  int head_id = target_id % num_head;
  int col = (threadIdx.y << 5) + lane;  // [0, INFEATURE]
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float rs = 0.0f;
  int theidx;
  float theval;
  int jlimit;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
      theval = edge_value[(i + lane) * num_head + head_id];
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
  if (col < INFEATURE) vout[target_id * INFEATURE + col] = rs;
}

__global__ void gen_bwd_mean(
    Index *ptr, Index *idx, float *grads_in, float *vout_fwd, float *grads_out,
    int num_node,
    int INFEATURE)  // push the gradient to the neighbor vertex
{
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  int num_neighbor = end - begin;
  float grad = 0.0f;
  if (col < INFEATURE && num_neighbor > 0) {
    grad = grads_in[row * INFEATURE + col] / num_neighbor;
  }
  int theidx, jlimit;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane] * INFEATURE;
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      if (col < INFEATURE) {
        atomicAdd(grads_out + neighbor_id + col, grad);
      }
    }
  }
}

__global__ void gen_bwd_sum(
    Index *ptr, Index *idx, float *grads_in, float *vout_fwd, float *grads_out,
    int num_node,
    int INFEATURE)  // push the gradient to the neighbor vertex
{
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float grad = 0.0f;
  if (col < INFEATURE) {
    grad = grads_in[row * INFEATURE + col];
  }
  int theidx, jlimit;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane] * INFEATURE;
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      if (col < INFEATURE) {
        atomicAdd(grads_out + neighbor_id + col, grad);
      }
    }
  }
}

__global__ void gen_bwd_mean_edge_value(
    Index *ptr, Index *idx, float *edge_value, float *grads_in, float *vout_fwd,
    float *grads_out, int num_node,
    int INFEATURE)  // push the gradient to the neighbor vertex
{
  assert(false);
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  int num_neighbor = end - begin;
  float grad = 0.0f;
  if (col < INFEATURE && num_neighbor > 0) {
    grad = grads_in[row * INFEATURE + col] / num_neighbor;
  }
  int theidx, jlimit;
  float theval;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane] * INFEATURE;
      theval = edge_value[i + lane];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      float val = __shfl_sync(0xffffffff, theval, j, 32);
      if (col < INFEATURE) {
        atomicAdd(grads_out + neighbor_id + col, grad * val);
      }
    }
  }
}

__global__ void gen_bwd_sum_edge_value(
    Index *ptr, Index *idx, float *edge_value, float *grads_in, float *vout_fwd,
    float *grads_out, int num_node,
    int INFEATURE)  // push the gradient to the neighbor vertex
{
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float grad = 0.0f;
  if (col < INFEATURE) {
    grad = grads_in[row * INFEATURE + col];
  }
  int theidx, jlimit;
  float theval;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane] * INFEATURE;
      theval = edge_value[i + lane];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      float val = __shfl_sync(0xffffffff, theval, j, 32);
      if (col < INFEATURE) {
        atomicAdd(grads_out + neighbor_id + col, grad * val);
      }
    }
  }
}

__global__ void gen_bwd_sum_edge_value_edge_grad(
    Index *ptr, Index *idx, float *edge_value, float *grads_in, float *vout_fwd,
    float *grads_out, int num_node, int INFEATURE,
    float *edge_grad)  // push the gradient to the neighbor vertex
{
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float grad = 0.0f;
  if (col < INFEATURE) {
    grad = grads_in[row * INFEATURE + col];
  }
  int theidx, jlimit;
  float theval;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
      theval = edge_value[i + lane];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      float val = __shfl_sync(0xffffffff, theval, j, 32);
      float tmp_edge_grad = 0.0f;
      if (col < INFEATURE) {
        atomicAdd(grads_out + neighbor_id * INFEATURE + col, grad * val);
        tmp_edge_grad = grad * vout_fwd[neighbor_id * INFEATURE + col];
      }
      for (int k = 16; k > 0; k >>= 1) {
        tmp_edge_grad += __shfl_down_sync(0xffffffff, tmp_edge_grad, k);  // sum
      }
      if (lane == 0) {
        atomicAdd(edge_grad + i + j, tmp_edge_grad);
      }
    }
  }
}

__global__ void gen_bwd_mean_edge_value_multi_head(
    Index *ptr, Index *idx, float *edge_value, float *grads_in, float *vout_fwd,
    float *grads_out, int num_node, int INFEATURE,
    int num_head)  // push the gradient to the neighbor vertex
{
  assert(false);
  int lane = threadIdx.x & 31;
  int target_id = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int row = target_id / num_head;
  int head_id = target_id % num_head;
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  int num_neighbor = end - begin;
  float grad = 0.0f;
  if (col < INFEATURE && num_neighbor > 0) {
    grad = grads_in[target_id * INFEATURE + col] / num_neighbor;
  }
  int theidx, jlimit;
  float theval;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
      theval = edge_value[(i + lane) * num_head + head_id];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      float val = __shfl_sync(0xffffffff, theval, j, 32);
      if (col < INFEATURE) {
        atomicAdd(
            grads_out + (neighbor_id * num_head + head_id) * INFEATURE + col,
            grad * val);
      }
    }
  }
}

__global__ void gen_bwd_sum_edge_value_multi_head(
    Index *ptr, Index *idx, float *edge_value, float *grads_in, float *vout_fwd,
    float *grads_out, int num_node, int INFEATURE,
    int num_head)  // push the gradient to the neighbor vertex
{
  int lane = threadIdx.x & 31;
  int target_id = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int row = target_id / num_head;
  int head_id = target_id % num_head;
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float grad = 0.0f;
  if (col < INFEATURE) {
    grad = grads_in[target_id * INFEATURE + col];
  }
  int theidx, jlimit;
  float theval;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
      theval = edge_value[(i + lane) * num_head + head_id];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      float val = __shfl_sync(0xffffffff, theval, j, 32);
      if (col < INFEATURE) {
        // printf("neighbor_id %d col %d grad %f val %f \n", neighbor_id, col,
        // grad, val);
        atomicAdd(
            grads_out + (neighbor_id * num_head + head_id) * INFEATURE + col,
            grad * val);
      }
    }
  }
}

__global__ void gen_bwd_sum_edge_value_multi_head_edge_grad(
    Index *ptr, Index *idx, float *edge_value, float *grads_in, float *vout_fwd,
    float *grads_out, int num_node, int INFEATURE, int num_head,
    float *edge_grad)  // push the gradient to the neighbor vertex
{
  int lane = threadIdx.x & 31;
  int target_id = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int row = target_id / num_head;
  int head_id = target_id % num_head;
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float grad = 0.0f;
  if (col < INFEATURE) {
    grad = grads_in[target_id * INFEATURE + col];
  }
  int theidx, jlimit;
  float theval;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
      theval = edge_value[(i + lane) * num_head + head_id];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      float val = __shfl_sync(0xffffffff, theval, j, 32);
      float tmp_edge_grad = 0.f;
      if (col < INFEATURE) {
        // printf("neighbor_id %d col %d grad %f val %f \n", neighbor_id, col,
        // grad, val);
        atomicAdd(
            grads_out + (neighbor_id * num_head + head_id) * INFEATURE + col,
            grad * val);
        tmp_edge_grad =
            grad *
            vout_fwd[(neighbor_id * num_head + head_id) * INFEATURE + col];
      }
      for (int k = 16; k > 0; k >>= 1) {
        tmp_edge_grad += __shfl_down_sync(0xffffffff, tmp_edge_grad, k);  // sum
      }
      if (lane == 0) {
        atomicAdd(edge_grad + (i + j) * num_head + head_id, tmp_edge_grad);
      }
    }
  }
}

__global__ void selective_aggr_fwd_kernel(Index *ptr, Index *idx, float *vin,
                                          float *vout, bool *mask, int num_node,
                                          int INFEATURE) {
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float rs = 0.0f;
  int theidx;
  int jlimit;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      if (mask[i + lane])
        theidx = idx[i + lane];
      else
        theidx = -1;
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      if (neighbor_id != -1 && col < INFEATURE)
        rs += vin[neighbor_id * INFEATURE + col];
    }
  }
  if (col < INFEATURE) atomicAdd(vout + row * INFEATURE + col, rs);
}

__global__ void selective_aggr_bwd_kernel(
    Index *ptr, Index *idx, float *grads_in, float *grads_out, bool *mask,
    Index num_node,
    int INFEATURE)  // push the gradient to the neighbor vertex
{
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float grad = 0.0f;
  if (col < INFEATURE) {
    grad = grads_in[row * INFEATURE + col];
  }
  int theidx, jlimit;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      if (mask[i + lane])
        theidx = idx[i + lane] * INFEATURE;
      else
        theidx = -1;
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      if (neighbor_id != -1 && col < INFEATURE) {
        atomicAdd(grads_out + neighbor_id + col, grad);
      }
    }
  }
}

__global__ void target_aggr(Index *ptr, Index *idx, Index *targets, float *vin,
                            float *vout, int num_node, int INFEATURE) {
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1], target_id = targets[row];
  float rs = 0.0f;
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
      if (neighbor_id != -1 && col < INFEATURE)
        rs += vin[neighbor_id * INFEATURE + col];
    }
  }
  if (col < INFEATURE) atomicAdd(vout + target_id * INFEATURE + col, rs);
}

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