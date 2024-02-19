#include "rel_spmm.h"

// Method 1: Transform all input feature using weight matrix of all relations
// and then do aggregation according to relation on the edge

// Assuming the input (vin) is of shape (num_rel, num_node, INFEATURE)
// it is already transformed by weight matrix
__global__ void aggr_rel_fwd(Index *ptr, Index *idx, int *rel, float *vin,
                             float *vout, int num_node, int INFEATURE,
                             Index transform_size) {
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float rs = 0.0f;
  int theidx;
  int therel;
  int jlimit;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane];
      therel = rel[i + lane];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      int rel_id = __shfl_sync(0xffffffff, therel, j, 32);
      rs += vin[transform_size * rel_id + neighbor_id * INFEATURE + col];
    }
  }
  if (col < INFEATURE) vout[row * INFEATURE + col] = rs;
}

torch::Tensor AggrRelFunction::forward(AutogradContext *ctx,
                                       torch::Tensor input, torch::Tensor ptr,
                                       torch::Tensor idx, torch::Tensor rel,
                                       Index num_node, int num_rel) {
  ctx->save_for_backward({input, ptr, idx, rel});
  if (num_node == 0) num_node = ptr.sizes()[0] - 1;
  ctx->saved_data["num_node"] = (int64_t)num_node;
  ctx->saved_data["num_rel"] = (int64_t)num_rel;

  int feat_len = input.sizes()[1];
  ASSERT(input.device().index() >= 0);
  checkCudaErrors(cudaSetDevice(input.device().index()));
  auto output = input.new_zeros({num_node, feat_len});
  int block_size = 512;
  dim3 grid, block;
  int ceil_feat_len = ((feat_len + 31) / 32 * 32);
  block_size = std::max(block_size, ceil_feat_len);
  grid.x = (num_node + (block_size / ceil_feat_len) - 1) /
           (block_size / ceil_feat_len);
  block.y = ceil_feat_len / 32;
  block.x = (block_size + ceil_feat_len - 1) / ceil_feat_len * 32;
  Index transform_size = (input.sizes()[0] / num_rel) * input.sizes()[1];
  aggr_rel_fwd<<<grid, block>>>(ptr.data<Index>(), idx.data<Index>(),
                                rel.data<int>(), input.data<float>(),
                                output.data<float>(), num_node, feat_len,
                                transform_size);
  return output;
}

torch::Tensor aggr_rel(torch::Tensor input, torch::Tensor ptr,
                       torch::Tensor idx, torch::Tensor etype, Index num_node,
                       int num_rel) {
  return AggrRelFunction::apply(input, ptr, idx, etype, num_node, num_rel);
}

__global__ void aggr_rel_bwd(
    Index *ptr, Index *idx, int *rel, float *grads_in, float *vout_fwd,
    float *grads_out, int num_node, int INFEATURE,
    Index transform_size)  // push the gradient to the neighbor vertex
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
  int theidx, jlimit, therel;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane] * INFEATURE;
      therel = rel[i + lane];
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
      int neighbor_id = __shfl_sync(0xffffffff, theidx, j, 32);
      int rel_id = __shfl_sync(0xffffffff, therel, j, 32);
      if (col < INFEATURE) {
        atomicAdd(grads_out + rel_id * transform_size + neighbor_id + col,
                  grad);
      }
    }
  }
}

tensor_list AggrRelFunction::backward(AutogradContext *ctx,
                                      tensor_list grad_outputs) {
  auto saved = ctx->get_saved_variables();
  auto input = saved[0];
  auto ptr = saved[1];
  auto idx = saved[2];
  auto rel = saved[3];
  auto grad_output = grad_outputs[0];
  auto grad_input = torch::zeros_like(input);

  int num_node = ctx->saved_data["num_node"].toInt();
  int num_rel = ctx->saved_data["num_rel"].toInt();
  int feat_len = input.sizes()[1];
  int ceil_feat_len = ((feat_len + 31) / 32 * 32);
  int block_size = 512;
  block_size = std::max(block_size, ceil_feat_len);

  dim3 grid, block;
  grid.x = (num_node + (block_size / ceil_feat_len) - 1) /
           (block_size / ceil_feat_len);
  block.y = ceil_feat_len / 32;
  block.x = (block_size + ceil_feat_len - 1) / ceil_feat_len * 32;
  ASSERT(block.x % 32 == 0);
  Index transform_size = (input.sizes()[0] / num_rel) * input.sizes()[1];
  aggr_rel_bwd<<<grid, block>>>(ptr.data<Index>(), idx.data<Index>(),
                                rel.data<int>(), grad_output.data<float>(),
                                input.data<float>(), grad_input.data<float>(),
                                num_node, feat_len, transform_size);
  return {grad_input,      torch::Tensor(), torch::Tensor(), torch::Tensor(),
          torch::Tensor(), torch::Tensor(), torch::Tensor()};
}

__global__ void aggr_rgcn_direct_kernel(Index *ptr, Index *idx, float *vin,
                                        float *vout, Index num_v, int INFEATURE,
                                        int OUTFEATURE, int num_rel,
                                        float *weight, int *etype) {
  int block_size = blockDim.x * blockDim.y;
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= num_v) return;
  int lane = threadIdx.x;
  int warpid = threadIdx.y;

  const int begin = ptr[row], end = ptr[row + 1];
  const int whichv_fea = row * OUTFEATURE;

  extern __shared__ float sh_rgcn[];
  int *shared_idx = (int *)(sh_rgcn + warpid * 32);
  int *shared_etype = (int *)(sh_rgcn + block_size + warpid * 32);
  // assuming kInFeatLen % 32 == 0
  int intimes = (INFEATURE + 31) / 32;
  int outtimes = (OUTFEATURE + 31) / 32;
  for (int cas = 0; cas < outtimes; cas++) {
    // working on col : cas * 32 + lane
    int thisoutfea = cas * 32 + lane;
    float rs = 0;
    for (int i = begin; i < end; i += 32) {
      shared_idx[lane] = idx[i + lane] * INFEATURE;
      shared_etype[lane] = etype[i + lane] * INFEATURE * OUTFEATURE;
      int jlimit = 32, j = 0;
      if (i + 32 >= end) jlimit = end - i;
      for (j = 0; j < jlimit; j++) {
        // TILING 1
        int theweight = shared_etype[j];
        for (int t = 0; t < intimes; t++) {
          float val = vin[shared_idx[j] + t * 32 + lane];
          for (int k = 0; k < 32; k++) {
            if (k + t * 32 < INFEATURE)
              rs += __shfl_sync(0xffffffff, val, k, 32) *
                    weight[theweight + (t * 32 + k) * OUTFEATURE + thisoutfea];
          }
        }
      }
    }
    if (thisoutfea < OUTFEATURE) atomicAdd(&vout[whichv_fea + thisoutfea], rs);
  }
}

__global__ void aggr_rgcn_direct_bwd_kernel(Index *ptr, Index *idx,
                                            float *vout_grad, Index num_v,
                                            int INFEATURE, int OUTFEATURE,
                                            int num_rel, float *weight,
                                            float *output_vin_grad,
                                            int *etype) {
  int block_size = blockDim.x * blockDim.y;
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= num_v) return;
  int lane = threadIdx.x;
  int warpid = threadIdx.y;

  const int begin = ptr[row], end = ptr[row + 1];
  const int whichv_fea = row * OUTFEATURE;

  extern __shared__ float sh_rgcn[];
  int *shared_idx = (int *)(sh_rgcn + warpid * 32);
  int *shared_etype = (int *)(sh_rgcn + block_size + warpid * 32);
  int intimes = (INFEATURE + 31) / 32;
  int outtimes = (OUTFEATURE + 31) / 32;
  for (int cas = 0; cas < intimes; cas++) {
    // working on col : cas * 32 + lane
    int thisinfea = cas * 32 + lane;
    float rs = 0;
    for (int i = begin; i < end; i += 32) {
      shared_idx[lane] = idx[i + lane] * INFEATURE;
      shared_etype[lane] = etype[i + lane] * INFEATURE * OUTFEATURE;
      int jlimit = 32, j = 0;
      if (i + 32 >= end) jlimit = end - i;
      for (j = 0; j < jlimit; j++) {
        // TILING 1
        int theweight = shared_etype[j];
        for (int t = 0; t < outtimes; t++) {
          float val = vout_grad[whichv_fea + t * 32 + lane];
          for (int k = 0; k < 32; k++) {
            if (k + t * 32 < OUTFEATURE)
              rs += __shfl_sync(0xffffffff, val, k, 32) *
                    weight[theweight + (t * 32 + k) + thisinfea * OUTFEATURE];
          }
        }
        if (thisinfea < INFEATURE)
          atomicAdd(&output_vin_grad[shared_idx[j] * INFEATURE + thisinfea],
                    rs);
      }
    }
  }
}

__global__ void typed_linear_kernel(float *vin, float *weights, float *vout,
                                    Index num, int INFEATURE, int OUTFEATURE,
                                    int in_feat_tile, int *types) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  if (warp_id >= num) return;
  extern __shared__ float sh[];
  int sh_mem_offset = in_feat_tile * (threadIdx.x / 32);
  float *sh_vin = sh + sh_mem_offset;
  float *curr_weight = weights + INFEATURE * OUTFEATURE * types[warp_id];
  for (int i = 0; i < INFEATURE / in_feat_tile; ++i) {
    // load vin into shared memory
    for (int j = lane_id; j < in_feat_tile; j += 32) {
      sh_vin[j] = vin[warp_id * INFEATURE + i * in_feat_tile + j];
    }
    for (int j = lane_id; j < OUTFEATURE; j += 32) {
      float res = 0.f;
      for (int k = 0; k < in_feat_tile; ++k) {
        res += sh_vin[k] * curr_weight[(i * in_feat_tile + k) * OUTFEATURE + j];
      }
      vout[warp_id * OUTFEATURE + j] += res;
    }
  }
}

void run_typed_linear(Tensor vin, Tensor weights, Tensor output, Tensor types,
                      int in_feat_tile) {
  int num = vin.size(0);
  int in_feat = vin.size(1);
  int out_feat = output.size(1);
  int block_size = 256;
  int num_blocks = ceil(num, block_size / 32);
  if (in_feat_tile > in_feat) in_feat_tile = in_feat;
  typed_linear_kernel<<<num_blocks, block_size,
                        in_feat_tile * 32 * sizeof(float)>>>(
      vin.data<float>(), weights.data<float>(), output.data<float>(), num,
      in_feat, out_feat, in_feat_tile, types.data<int>());
}

__global__ void typed_linear_kernel_s2e(float *vin, float *weights,
                                        Index *src_ids, float *vout, Index num,
                                        int INFEATURE, int OUTFEATURE,
                                        int in_feat_tile, int *types) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  if (warp_id >= num) return;
  Index src_id = src_ids[warp_id];
  extern __shared__ float sh[];
  int sh_mem_offset = in_feat_tile * (threadIdx.x / 32);
  float *sh_vin = sh + sh_mem_offset;
  float *curr_weight = weights + INFEATURE * OUTFEATURE * types[warp_id];
  for (int i = 0; i < INFEATURE / in_feat_tile; ++i) {
    // load vin into shared memory
    for (int j = lane_id; j < in_feat_tile; j += 32) {
      sh_vin[j] = vin[src_id * INFEATURE + i * in_feat_tile + j];
    }
    for (int j = lane_id; j < OUTFEATURE; j += 32) {
      float res = 0.f;
      for (int k = 0; k < in_feat_tile; ++k) {
        res += sh_vin[k] * curr_weight[(i * in_feat_tile + k) * OUTFEATURE + j];
      }
      vout[warp_id * OUTFEATURE + j] += res;
    }
  }
}

void run_typed_linear_s2e(Tensor vin, Tensor weights, Tensor output,
                          Tensor src_id, Tensor types, int in_feat_tile) {
  int num = src_id.size(0);
  int in_feat = vin.size(1);
  int out_feat = output.size(1);
  int block_size = 256;
  int num_blocks = ceil(num, block_size / 32);
  if (in_feat_tile > in_feat) in_feat_tile = in_feat;
  typed_linear_kernel_s2e<<<num_blocks, block_size,
                            in_feat_tile * 32 * sizeof(float)>>>(
      vin.data<float>(), weights.data<float>(), src_id.data<Index>(),
      output.data<float>(), num, in_feat, out_feat, in_feat_tile,
      types.data<int>());
}

__global__ void typed_linear_kernel_s2d(float *vin, float *weights,
                                        Index *src_ids, Index *dst_ids,
                                        float *vout, Index num, int INFEATURE,
                                        int OUTFEATURE, int in_feat_tile,
                                        int *types) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  if (warp_id >= num) return;
  Index src_id = src_ids[warp_id];
  Index dst_id = dst_ids[warp_id];
  extern __shared__ float sh[];
  int sh_mem_offset = in_feat_tile * (threadIdx.x / 32);
  float *sh_vin = sh + sh_mem_offset;
  float *curr_weight = weights + INFEATURE * OUTFEATURE * types[warp_id];
  for (int i = 0; i < INFEATURE / in_feat_tile; ++i) {
    // load vin into shared memory
    for (int j = lane_id; j < in_feat_tile; j += 32) {
      sh_vin[j] = vin[src_id * INFEATURE + i * in_feat_tile + j];
    }
    for (int j = lane_id; j < OUTFEATURE; j += 32) {
      float res = 0.f;
      for (int k = 0; k < in_feat_tile; ++k) {
        res += sh_vin[k] * curr_weight[(i * in_feat_tile + k) * OUTFEATURE + j];
      }
      atomicAdd(vout + dst_id * OUTFEATURE + j, res);
    }
  }
}

void run_typed_linear_s2d(Tensor vin, Tensor weights, Tensor output,
                          Tensor src_id, Tensor dst_id, Tensor types,
                          int in_feat_tile) {
  int num = src_id.size(0);
  int in_feat = vin.size(1);
  int out_feat = output.size(1);
  int block_size = 256;
  int num_blocks = ceil(num, block_size / 32);
  if (in_feat_tile > in_feat) in_feat_tile = in_feat;
  typed_linear_kernel_s2d<<<num_blocks, block_size,
                            in_feat_tile * 32 * sizeof(float)>>>(
      vin.data<float>(), weights.data<float>(), src_id.data<Index>(),
      dst_id.data<Index>(), output.data<float>(), num, in_feat, out_feat,
      in_feat_tile, types.data<int>());
}

torch::Tensor AggrRelDirectFunction::forward(
    AutogradContext *ctx, torch::Tensor input, torch::Tensor ptr,
    torch::Tensor idx, torch::Tensor weights, torch::Tensor rel, Index num_node,
    int num_rel) {
  ctx->save_for_backward({input, weights});

  ctx->saved_data["ptr"] = (int64_t)ptr.data<Index>();
  ctx->saved_data["idx"] = (int64_t)idx.data<Index>();
  ctx->saved_data["rel"] = (int64_t)rel.data<int>();
  if (num_node == 0) num_node = ptr.sizes()[0] - 1;
  ctx->saved_data["num_node"] = (int64_t)num_node;
  ctx->saved_data["num_rel"] = (int64_t)num_rel;

  // weight shape = [rel, in_feat, out_feat]
  int feat_len = input.sizes()[1];
  int out_feat_len = weights.sizes()[2];
  ASSERTWITH(feat_len == weights.sizes()[1], "feat_len weight size {} {}",
             feat_len, weights.sizes()[1]);
  ASSERT(feat_len % 32 == 0);
  // ASSERT(out_feat_len % 32 == 0);
  ASSERT(input.device().index() >= 0);
  checkCudaErrors(cudaSetDevice(input.device().index()));
  // auto output = input.new_zeros({input.sizes()[0] / num_rel, feat_len});
  // ASSERT(output.device().index() == input.device().index());
  // output.requires_grad_(true);
  auto output = input.new_zeros({num_node, out_feat_len});
  int block_size = 256;
  int tmp_target_in_block = block_size / 32;
  dim3 grid, block;
  grid.x = (num_node + tmp_target_in_block - 1) / tmp_target_in_block;
  block.x = 32;
  block.y = block_size / 32;
  int shared_size = block_size * 2 * sizeof(int);
  aggr_rgcn_direct_kernel<<<grid, block, shared_size>>>(
      ptr.data<Index>(), idx.data<Index>(), input.data<float>(),
      output.data<float>(), num_node, feat_len, out_feat_len, num_rel,
      weights.data<float>(), rel.data<int>());
  return output;
}

tensor_list AggrRelDirectFunction::backward(AutogradContext *ctx,
                                            tensor_list grad_outputs) {
  auto saved = ctx->get_saved_variables();
  auto input = saved[0];
  auto weights = saved[1];
  Index *ptr = (Index *)(ctx->saved_data["ptr"].toInt());
  Index *idx = (Index *)(ctx->saved_data["idx"].toInt());
  int *rel = (int *)(ctx->saved_data["rel"].toInt());
  auto grad_output = grad_outputs[0];
  auto grad_input = torch::zeros_like(input);
  auto grad_weight = torch::zeros_like(weights);
  int num_node = ctx->saved_data["num_node"].toInt();
  int num_rel = ctx->saved_data["num_rel"].toInt();
  int feat_len = input.sizes().back();  // input: [nodes, heads, channels]
  int out_feat_len = weights.sizes().back();

  int block_size = 256;
  int tmp_target_in_block = block_size / 32;
  dim3 grid, block;
  grid.x = (num_node + tmp_target_in_block - 1) / tmp_target_in_block;
  block.x = 32;
  block.y = block_size / 32;
  int shared_size = block_size * 2 * sizeof(int);
  // aggr_rgcn_direct_bwd_kernel<<< grid, block, shared_size >>>(ptr, idx,
  //                             grad_output.data<float>(), num_node, feat_len,
  //                             out_feat_len, num_rel,
  //                             weights.data<float>(),
  //                             grad_input.data<float>(),
  //                             rel);

  return {grad_input,      torch::Tensor(), torch::Tensor(), grad_weight,
          torch::Tensor(), torch::Tensor(), torch::Tensor()};
}

torch::Tensor aggr_rel_direct(torch::Tensor input, torch::Tensor ptr,
                              torch::Tensor idx, torch::Tensor weights,
                              torch::Tensor etype, Index num_node,
                              int num_rel) {
  return AggrRelDirectFunction::apply(input, ptr, idx, weights, etype, num_node,
                                      num_rel);
}

torch::Tensor aggr_rgcn_direct_func(torch::Tensor input, torch::Tensor ptr,
                                    torch::Tensor idx, torch::Tensor weights,
                                    torch::Tensor rel, Index num_node) {
  int num_rel = weights.sizes()[0];
  int feat_len = input.sizes()[1];
  int out_feat_len = weights.sizes()[2];
  ASSERTWITH(feat_len == weights.sizes()[1], "feat_len weight size {} {}",
             feat_len, weights.sizes()[1]);
  // ASSERT(feat_len % 32 == 0);
  // ASSERT(out_feat_len % 32 == 0);
  ASSERT(input.device().index() >= 0);
  ASSERT(weights.dim() == 3);
  auto output = input.new_zeros({num_node, out_feat_len});
  int block_size = 256;
  int tmp_target_in_block = block_size / 32;
  dim3 grid, block;
  grid.x = (num_node + tmp_target_in_block - 1) / tmp_target_in_block;
  block.x = 32;
  block.y = block_size / 32;
  int shared_size = block_size * 2 * sizeof(int);
  aggr_rgcn_direct_kernel<<<grid, block, shared_size>>>(
      ptr.data<Index>(), idx.data<Index>(), input.data<float>(),
      output.data<float>(), num_node, feat_len, out_feat_len, num_rel,
      weights.data<float>(), rel.data<int>());
  return output;
}

__global__ void RgcnLayer1KernelImpl(const Index* ranges, 
  const Index* src_ids, 
  const Index* types, 
  const float* hidden, 
  const float* weight, 
  float* ret, 
  Index num_nodes, 
  Index feat_len_y, 
  Index feat_len_x, 
  Index ntypes) {
    Index tx = threadIdx.x + (blockIdx.x % feat_len_x) * feat_len_y;
    if (blockIdx.x < num_nodes) {
        Index beg = __ldg(ranges + blockIdx.x);
        Index end = __ldg(ranges + blockIdx.x + 1);
        Index tx = threadIdx.x;
        for (;tx<feat_len_x * feat_len_y; tx += blockDim.x) {
          Index ty = tx / feat_len_x;
          Index th = tx % feat_len_x;
          float agg_val = 0.; 
          float w = 0.;
          Index cur_type_id = -1;
          for(;beg<end;beg++) {
              Index src_id = __ldg(src_ids + beg);
              // Index eid = __ldg(eids + beg);
              Index type_id = __ldg(types + beg);
              if (type_id != cur_type_id) {
                  w = __ldg(weight + type_id*feat_len_y*feat_len_x + tx);
              }
              float h = __ldg(hidden + src_id*feat_len_y + ty);
              // float n = __ldg(norm + eid);
              agg_val += h * w;
          }
          atomicAdd(ret + blockIdx.x*feat_len_x + th, agg_val);
      }
    }
}

torch::Tensor call_seastar_rgcn_forward(torch::Tensor input, torch::Tensor ptr,
                              torch::Tensor idx, torch::Tensor weights,
                              torch::Tensor etype) {
  Index ntypes = weights.sizes()[0];
  Index feat_len_y = weights.sizes()[1];
  Index feat_len_x = weights.sizes()[2];
  int nblks = ptr.sizes()[0] - 1;
  int nthrs = feat_len_y;
  auto output = input.new_zeros({ptr.sizes()[0] - 1, feat_len_x});
  RgcnLayer1KernelImpl<<<nblks, nthrs >>>
      (ptr.data<Index>(), idx.data<Index>(), etype.data<Index>(), input.data<float>(),
      weights.data<float>(), output.data<float>(), ptr.sizes()[0] - 1, feat_len_y, feat_len_x, ntypes);
  checkCudaErrors(cudaDeviceSynchronize());
  return output;
}


__global__ void RgcnLayer1BackwardKernelImpl(Index* ranges, 
  Index* dst_ids, 
  Index* types, 
  float* hidden, 
  float* weight, 
  float* grad_out, 
  float* grad_hidden, 
  float* grad_weight, 
  Index num_nodes, 
  Index feat_len_y, 
  Index feat_len_x, 
  Index ntypes) {
    if (blockIdx.x < num_nodes) {
        Index beg = __ldg(ranges + blockIdx.x);
        Index end = __ldg(ranges + blockIdx.x + 1);
        Index tx = threadIdx.x;
        for (;tx<feat_len_x * feat_len_y; tx += blockDim.x) {
            Index ty = tx / feat_len_x;
            Index th = tx % feat_len_x;
            float h = __ldg(hidden + blockIdx.x*feat_len_y + ty);
            float agg = 0.;
            for(;beg<end;beg++) {
                Index dst_id = __ldg(dst_ids + beg);
                // Index eid = __ldg(eids + beg);
                Index type_id = __ldg(types + beg);
                float g = __ldg(grad_out + dst_id * feat_len_x + th);
                float w = __ldg(weight + type_id*feat_len_y*feat_len_x + tx);
                // float n = __ldg(norm + eid);
                agg += g*w;
                atomicAdd(grad_weight + type_id*feat_len_y*feat_len_x + tx, g*h);
            }
            atomicAdd(grad_hidden + blockIdx.x*feat_len_y + ty, agg);
        }
    }
}

std::vector<torch::Tensor> call_seastar_rgcn_backward(torch::Tensor input, torch::Tensor ptr,
                              torch::Tensor idx, torch::Tensor weights,
                              torch::Tensor etype, torch::Tensor grad) {
  auto grad_input = torch::zeros_like(input);
  auto grad_weight = torch::zeros_like(weights);
  Index ntypes = weights.sizes()[0];
  Index feat_len_y = weights.sizes()[1];
  Index feat_len_x = weights.sizes()[2];

  int nblks = ptr.sizes()[0] - 1;
  int nthrs = feat_len_y;

  RgcnLayer1BackwardKernelImpl<<<nblks, nthrs>>> (
    ptr.data<Index>(), idx.data<Index>(), etype.data<Index>(), input.data<float>(),
    weights.data<float>(),
    grad.data<float>(),
    grad_input.data<float>(),
    grad_weight.data<float>(),
    ptr.sizes()[0] - 1, feat_len_y, feat_len_x, ntypes);
    return {grad_input, grad_weight};
}