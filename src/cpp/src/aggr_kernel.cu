#include "aggr_kernel.h"

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

__global__ void target_aggr_backward(Index *ptr, Index *idx, Index *targets, const float *grads_in,
                            float *grads_out, int num_node, int INFEATURE) {
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  if (col >= INFEATURE) return;
  Index begin = ptr[row], end = ptr[row + 1], target_id = targets[row];
  float grad = grads_in[target_id * INFEATURE + col];
  int theidx, jlimit;
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
        // rs += vin[neighbor_id * INFEATURE + col];
        atomicAdd(grads_out + neighbor_id * INFEATURE + col, grad);
    }
  }
}