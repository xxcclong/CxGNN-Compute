#include "aggr.h"
#include "aggr_kernel.h"
#include "common.h"

__global__ void gen_fwd_with_target(Index *ptr, Index *idx, Index *targets,
                                    float *val, float *vin, float *vout,
                                    int num_target, int INFEATURE) {
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_target) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float rs = 0.0f;
  // // int theidx;
  // float theval;
  int jlimit;
#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    //         if (i + lane < end)
    //         {
    //             theidx = idx[i + lane] * INFEATURE;
    // #ifdef WITHVALUE
    //             theval = val[i + lane];
    // #endif
    //         }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
#ifdef WITHVALUE
      // rs += vin[__shfl(theidx, j, 32) + col] * __shfl(theval, j, 32);
      rs += vin[idx[j + i] * INFEATURE + col] * val[j + i];
#else
      // rs += vin[__shfl(theidx, j, 32) + col];
      rs += vin[idx[j + i] * INFEATURE + col];
#endif
    }
  }
  if (col < INFEATURE)
    vout[targets[row] * INFEATURE + col] = max(rs, 0.f);  // relu
}

__global__ void gather_fwd(float *vin, Index *dest, float *vout, int num_edge,
                           int feat_len) {
  int lane = threadIdx.x & 31;
  int eid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int col = (threadIdx.y << 5) + lane;
  if (eid >= num_edge) return;
  Index vid = dest[eid];
  if (col < feat_len)
    atomicAdd(vout + vid * feat_len + col, vin[eid * feat_len + col]);
}

__global__ void gather_bwd(float *grad_vout, Index *dest, float *grad_vin,
                           int num_edge, int feat_len) {
  int lane = threadIdx.x & 31;
  int eid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int col = (threadIdx.y << 5) + lane;
  if (eid >= num_edge) return;
  Index vid = dest[eid];
  if (col < feat_len)
    grad_vin[eid * feat_len + col] = grad_vout[vid * feat_len + col];
}

__global__ void gen_bwd_with_target(
    Index *ptr, Index *idx, Index *targets, float *val, float *grads_in,
    float *vout_fwd, float *grads_out, int num_target,
    int INFEATURE)  // push the gradient to the neighbor vertex
{
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  Index target = targets[row];
  int col = (threadIdx.y << 5) + lane;
  if (row >= num_target) return;
  Index begin = ptr[row], end = ptr[row + 1];
  float grad = 0.0f;
  if (col < INFEATURE && vout_fwd[target * INFEATURE + col] > 0.f) {
    grad = grads_in[target * INFEATURE + col];  // gradient for relu operator
  }
  int theidx;
  int jlimit;
#ifdef WITHVALUE
  float theval;
#endif

#pragma unroll
  for (Index i = begin; i < end; i += 32) {
    if (i + lane < end) {
      theidx = idx[i + lane] * INFEATURE;
#ifdef WITHVALUE
      theval = val[i + lane];
#endif
    }
    jlimit = 32;
    if (end - i < 32) jlimit = end - i;
    for (int j = 0; j < jlimit; ++j) {
#ifdef WITHVALUE
      // atomicAdd()
      // rs += vin[__shfl(theidx, j, 32) + col] * __shfl(theval, j, 32);
#else
      // rs += vin[__shfl(theidx, j, 32) + col];
      atomicAdd(grads_out + __shfl_sync(0xffffffff, theidx, j, 32) + col, grad);
#endif
    }
  }
}

torch::Tensor sageForwardImpl(AutogradContext *ctx, torch::Tensor input,
                              torch::Tensor ptr, torch::Tensor idx,
                              int num_node, AggrType aggr_type,
                              torch::Tensor edge_value = torch::Tensor()) {
  if (edge_value.sizes()[0] == 0) {
    ctx->save_for_backward({input});
  } else {
    ctx->save_for_backward({input, edge_value});
  }
  ctx->saved_data["ptr"] = (int64_t)ptr.data<Index>();
  ctx->saved_data["idx"] = (int64_t)idx.data<Index>();
  if (num_node == 0) num_node = ptr.sizes()[0] - 1;
  ctx->saved_data["num_node"] = (int64_t)num_node;
  ctx->saved_data["aggr_type"] = (int64_t)aggr_type;

  int feat_len = input.sizes().back();  // input: [nodes, heads, channels]
  ASSERT(input.device().index() >= 0);
  checkCudaErrors(cudaSetDevice(input.device().index()));
  int num_head = 1;
  auto output = torch::Tensor();
  if (edge_value.sizes()[0] != 0 && edge_value.sizes().size() > 1) {
    num_head = edge_value.sizes()[1];
    ASSERT(num_head == input.sizes()[1]);
    output = input.new_zeros({num_node, num_head, feat_len});
  } else {
    output = input.new_zeros({num_node, feat_len});
  }
  // auto output = torch::zeros_like(input);
  output.requires_grad_(true);
  int block_size = 512;
  dim3 grid, block;
  int ceil_feat_len = ((feat_len + 31) / 32 * 32);
  block_size = std::max(block_size, ceil_feat_len);
  grid.x = (num_node + (block_size / ceil_feat_len) - 1) /
           (block_size / ceil_feat_len);
  block.y = ceil_feat_len / 32;
  block.x = (block_size + ceil_feat_len - 1) / ceil_feat_len * 32;

  if (edge_value.sizes()[0] == 0) {  // no edge value
    if (aggr_type == AggrType::Mean) {
      gen_fwd_mean<<<grid, block>>>(ptr.data<Index>(), idx.data<Index>(),
                                    input.data<float>(), output.data<float>(),
                                    num_node, feat_len);
    } else if (aggr_type == AggrType::Sum) {
      gen_fwd_sum<<<grid, block>>>(ptr.data<Index>(), idx.data<Index>(),
                                   input.data<float>(), output.data<float>(),
                                   num_node, feat_len);
    }
  } else {
    if (aggr_type == AggrType::Mean) {
      if (edge_value.sizes().size() == 1) {  // single head aggregation
        ASSERTWITH(0, "DO NOT use mean with edge value");
        gen_fwd_mean_edge_value<<<grid, block>>>(
            ptr.data<Index>(), idx.data<Index>(), edge_value.data<float>(),
            input.data<float>(), output.data<float>(), num_node, feat_len);
      } else {  // multi head aggregation
        ASSERTWITH(0, "DO NOT use mean with edge value");
        grid.x = (num_node * num_head + (block_size / ceil_feat_len) - 1) /
                 (block_size / ceil_feat_len);
        gen_fwd_mean_edge_value_multi_head<<<grid, block>>>(
            ptr.data<Index>(), idx.data<Index>(), edge_value.data<float>(),
            input.data<float>(), output.data<float>(), num_node, feat_len,
            num_head);
      }
    } else if (aggr_type == AggrType::Sum) {
      if (edge_value.sizes().size() == 1) {  // single head aggregation
        gen_fwd_sum_edge_value<<<grid, block>>>(
            ptr.data<Index>(), idx.data<Index>(), edge_value.data<float>(),
            input.data<float>(), output.data<float>(), num_node, feat_len);
      } else {  // multi head aggregation
        grid.x = (num_node * num_head + (block_size / ceil_feat_len) - 1) /
                 (block_size / ceil_feat_len);
        gen_fwd_sum_edge_value_multi_head<<<grid, block>>>(
            ptr.data<Index>(), idx.data<Index>(), edge_value.data<float>(),
            input.data<float>(), output.data<float>(), num_node, feat_len,
            num_head);
      }
    }
  }
  return output;
}

tensor_list sageBackwardImpl(AutogradContext *ctx, tensor_list grad_outputs) {
  auto saved = ctx->get_saved_variables();
  auto input = saved[0];
  Index *ptr = (Index *)(ctx->saved_data["ptr"].toInt());
  Index *idx = (Index *)(ctx->saved_data["idx"].toInt());
  AggrType aggr_type = (AggrType)(ctx->saved_data["aggr_type"].toInt());
  auto grad_output = grad_outputs[0];
  auto grad_input = torch::zeros_like(input);

  int num_node = ctx->saved_data["num_node"].toInt();
  int feat_len = input.sizes().back();  // input: [nodes, heads, channels]
  int ceil_feat_len = ((feat_len + 31) / 32 * 32);
  int block_size = 512;
  block_size = std::max(block_size, ceil_feat_len);

  dim3 grid, block;
  grid.x = (num_node + (block_size / ceil_feat_len) - 1) /
           (block_size / ceil_feat_len);
  block.y = ceil_feat_len / 32;
  block.x = (block_size + ceil_feat_len - 1) / ceil_feat_len * 32;
  ASSERT(block.x % 32 == 0);
  if (saved.size() == 1) {
    if (aggr_type == AggrType::Mean)
      gen_bwd_mean<<<grid, block>>>(
          ptr, idx, grad_output.data<float>(), input.data<float>(),
          grad_input.data<float>(), num_node, feat_len);
    else if (aggr_type == AggrType::Sum)
      gen_bwd_sum<<<grid, block>>>(
          ptr, idx, grad_output.data<float>(), input.data<float>(),
          grad_input.data<float>(), num_node, feat_len);
  } else {
    auto edge_value = saved[1];
    if (aggr_type == AggrType::Mean) {
      if (edge_value.sizes().size() == 1) {
        ASSERTWITH(0, "DO NOT use mean with edge value");
        gen_bwd_mean_edge_value<<<grid, block>>>(
            ptr, idx, edge_value.data<float>(), grad_output.data<float>(),
            input.data<float>(), grad_input.data<float>(), num_node, feat_len);
      } else {
        int num_head = edge_value.sizes()[1];
        grid.x = (num_node * num_head + (block_size / ceil_feat_len) - 1) /
                 (block_size / ceil_feat_len);
        ASSERTWITH(0, "DO NOT use mean with edge value");
        gen_bwd_mean_edge_value_multi_head<<<grid, block>>>(
            ptr, idx, edge_value.data<float>(), grad_output.data<float>(),
            input.data<float>(), grad_input.data<float>(), num_node, feat_len,
            num_head);
      }
    } else if (aggr_type == AggrType::Sum) {
      if (edge_value.sizes().size() == 1) {
        if (edge_value.requires_grad()) {
          auto edge_grad = torch::zeros_like(edge_value);
          gen_bwd_sum_edge_value_edge_grad<<<grid, block>>>(
              ptr, idx, edge_value.data<float>(), grad_output.data<float>(),
              input.data<float>(), grad_input.data<float>(), num_node, feat_len,
              edge_grad.data<float>());
          return {grad_input, edge_grad};
        } else {
          gen_bwd_sum_edge_value<<<grid, block>>>(
              ptr, idx, edge_value.data<float>(), grad_output.data<float>(),
              input.data<float>(), grad_input.data<float>(), num_node,
              feat_len);
        }
      } else {
        int num_head = edge_value.sizes()[1];
        grid.x = (num_node * num_head + (block_size / ceil_feat_len) - 1) /
                 (block_size / ceil_feat_len);
        if (edge_value.requires_grad()) {
          auto edge_grad = torch::zeros_like(edge_value);
          gen_bwd_sum_edge_value_multi_head_edge_grad<<<grid, block>>>(
              ptr, idx, edge_value.data<float>(), grad_output.data<float>(),
              input.data<float>(), grad_input.data<float>(), num_node, feat_len,
              num_head, edge_grad.data<float>());
          return {grad_input, edge_grad};
        } else {
          gen_bwd_sum_edge_value_multi_head<<<grid, block>>>(
              ptr, idx, edge_value.data<float>(), grad_output.data<float>(),
              input.data<float>(), grad_input.data<float>(), num_node, feat_len,
              num_head);
        }
      }
    }
  }
  return {grad_input};
}

torch::Tensor SAGEFunction::forward(AutogradContext *ctx, torch::Tensor input,
                                    torch::Tensor ptr, torch::Tensor idx,
                                    int num_node, AggrType aggr_type) {
  return sageForwardImpl(ctx, input, ptr, idx, num_node, aggr_type);
}

tensor_list SAGEFunction::backward(AutogradContext *ctx,
                                   tensor_list grad_outputs) {
  auto tl = sageBackwardImpl(ctx, grad_outputs);
  return {tl[0], torch::Tensor(), torch::Tensor(), torch::Tensor(),
          torch::Tensor()};
}

torch::Tensor SAGEEdgeValueFunction::forward(AutogradContext *ctx,
                                             torch::Tensor input,
                                             torch::Tensor ptr,
                                             torch::Tensor idx,
                                             torch::Tensor edge_value,
                                             int num_node, AggrType aggr_type) {
  return sageForwardImpl(ctx, input, ptr, idx, num_node, aggr_type, edge_value);
}

tensor_list SAGEEdgeValueFunction::backward(AutogradContext *ctx,
                                            tensor_list grad_outputs) {
  auto tl = sageBackwardImpl(ctx, grad_outputs);
  return {tl[0],           torch::Tensor(),
          torch::Tensor(), tl.size() == 2 ? tl[1] : torch::Tensor(),
          torch::Tensor(), torch::Tensor()};
}

// torch::Tensor AggrWithTargetFunction::forward(AutogradContext *ctx,
// torch::Tensor input,
//                                               torch::Tensor ptr,
//                                               torch::Tensor idx,
//                                               torch::Tensor targets, Index
//                                               num_node) {
//   ctx->save_for_backward({input});
//   ctx->saved_data["ptr"] = (int64_t)ptr.data<Index>();
//   ctx->saved_data["idx"] = (int64_t)idx.data<Index>();
//   ctx->saved_data["targets"] = (int64_t)targets.data<Index>();
//   int num_target = targets.sizes()[0];
//   ctx->saved_data["num_target"] = (int64_t)num_target;

//   int feat_len = input.sizes()[1];
//   auto output = input.new_zeros({num_target, feat_len});

//   // auto output = torch::empty_like(input);
//   output.requires_grad_(true);

//   int block_size = 512;

//   dim3 grid, block;
//   grid.x = (num_target + (block_size / ((feat_len + 31) / 32 * 32)) - 1) /
//            (block_size / ((feat_len + 31) / 32 * 32));
//   block.y = (feat_len + 31) / 32;
//   block.x = (block_size + block.y * 32 - 1) / (block.y * 32) * 32;

//   gen_fwd_with_target<<<grid, block>>>(ptr.data<Index>(), idx.data<Index>(),
//   targets.data<Index>(),
//                                        nullptr, input.data<float>(),
//                                        output.data<float>(), num_target,
//                                        feat_len);
//   return output;
// }

// tensor_list AggrWithTargetFunction::backward(AutogradContext *ctx,
// tensor_list grad_outputs) {
//   auto saved = ctx->get_saved_variables();
//   auto input = saved[0];
//   Index *ptr = (Index *)(ctx->saved_data["ptr"].toInt());
//   Index *idx = (Index *)(ctx->saved_data["idx"].toInt());
//   Index *targets = (Index *)(ctx->saved_data["targets"].toInt());
//   auto grad_output = grad_outputs[0];
//   auto grad_input = torch::zeros_like(input);

//   int num_target = ctx->saved_data["num_target"].toInt();
//   int feat_len = input.sizes()[1];
//   int block_size = 512;

//   dim3 grid, block;
//   grid.x = (num_target + (block_size / ((feat_len + 31) / 32 * 32)) - 1) /
//            (block_size / ((feat_len + 31) / 32 * 32));
//   block.y = (feat_len + 31) / 32;
//   block.x = (block_size + block.y * 32 - 1) / (block.y * 32) * 32;
//   ASSERT(block.x % 32 == 0);
//   gen_bwd_with_target<<<grid, block>>>(ptr, idx, targets, nullptr,
//   grad_output.data<float>(),
//                                        input.data<float>(),
//                                        grad_input.data<float>(), num_target,
//                                        feat_len);

//   return {grad_input, torch::Tensor(), torch::Tensor()};
// }

torch::Tensor sage_mean_forward_edge_value(torch::Tensor input,
                                           torch::Tensor ptr, torch::Tensor idx,
                                           torch::Tensor edge_value,
                                           int num_node) {
  return SAGEEdgeValueFunction::apply(input, ptr, idx, edge_value, num_node,
                                      AggrType::Mean);
}

torch::Tensor sage_sum_forward_edge_value(torch::Tensor input,
                                          torch::Tensor ptr, torch::Tensor idx,
                                          torch::Tensor edge_value,
                                          int num_node) {
  return SAGEEdgeValueFunction::apply(input, ptr, idx, edge_value, num_node,
                                      AggrType::Sum);
}

torch::Tensor sage_mean_forward(torch::Tensor input, torch::Tensor ptr,
                                torch::Tensor idx, int num_node) {
  return SAGEFunction::apply(input, ptr, idx, num_node, AggrType::Mean);
}

torch::Tensor sage_sum_forward(torch::Tensor input, torch::Tensor ptr,
                               torch::Tensor idx, int num_node) {
  return SAGEFunction::apply(input, ptr, idx, num_node, AggrType::Sum);
}

torch::Tensor GatherFunction::forward(AutogradContext *ctx, torch::Tensor input,
                                      torch::Tensor dest, Index num_node) {
  ctx->save_for_backward({input, dest});
  ASSERT(num_node > 0);

  int feat_len = input.sizes()[1];
  int num_edge = dest.sizes()[0];
  ASSERT(input.device().index() >= 0);
  checkCudaErrors(cudaSetDevice(input.device().index()));
  auto output = input.new_zeros({num_node, feat_len});
  output.requires_grad_(true);
  int block_size = 512;
  dim3 grid, block;
  int ceil_feat_len = ((feat_len + 31) / 32 * 32);
  block_size = std::max(block_size, ceil_feat_len);
  grid.x = (num_edge + (block_size / ceil_feat_len) - 1) /
           (block_size / ceil_feat_len);
  block.y = ceil_feat_len / 32;
  block.x = (block_size + ceil_feat_len - 1) / ceil_feat_len * 32;

  gather_fwd<<<grid, block>>>(input.data<float>(), dest.data<Index>(),
                              output.data<float>(), num_edge, feat_len);
  return output;
}

tensor_list GatherFunction::backward(AutogradContext *ctx,
                                     tensor_list grad_outputs) {
  auto saved = ctx->get_saved_variables();
  auto input = saved[0];
  auto dest = saved[1];
  auto grad_output = grad_outputs[0];
  auto grad_input = torch::empty_like(input);

  int num_edge = grad_input.sizes()[0];
  int feat_len = input.sizes()[1];
  int ceil_feat_len = ((feat_len + 31) / 32 * 32);
  int block_size = 512;
  block_size = std::max(block_size, ceil_feat_len);

  dim3 grid, block;
  grid.x = (num_edge + (block_size / ceil_feat_len) - 1) /
           (block_size / ceil_feat_len);
  block.y = ceil_feat_len / 32;
  block.x = (block_size + ceil_feat_len - 1) / ceil_feat_len * 32;
  ASSERT(block.x % 32 == 0);
  gather_bwd<<<grid, block>>>(grad_output.data<float>(), dest.data<Index>(),
                              grad_input.data<float>(), num_edge, feat_len);
  return {grad_input, torch::Tensor(), torch::Tensor()};
}

torch::Tensor gather(torch::Tensor input /* O(E) */,
                     torch::Tensor dest /* O(E) */, Index num_node) {
  return GatherFunction::apply(input, dest, num_node);
}

void target_sage_sum_forward(Tensor input, Tensor ptr, Tensor idx,
                             Tensor targets, Tensor output, int num_node) {
  int feat_len = input.sizes().back();
  int block_size = 512;
  dim3 grid, block;
  int ceil_feat_len = ((feat_len + 31) / 32 * 32);
  block_size = std::max(block_size, ceil_feat_len);
  grid.x = (num_node + (block_size / ceil_feat_len) - 1) /
           (block_size / ceil_feat_len);
  block.y = ceil_feat_len / 32;
  block.x = (block_size + ceil_feat_len - 1) / ceil_feat_len * 32;
  target_aggr<<<grid, block>>>(ptr.data<Index>(), idx.data<Index>(),
                               targets.data<Index>(), input.data<float>(),
                               output.data<float>(), num_node, feat_len);
}

void selective_aggr_fwd(Tensor input, Tensor ptr, Tensor idx, Tensor mask,
                        Tensor output, int num_node) {
  int feat_len = input.sizes().back();
  int block_size = 512;
  dim3 grid, block;
  int ceil_feat_len = ((feat_len + 31) / 32 * 32);
  block_size = std::max(block_size, ceil_feat_len);
  grid.x = (num_node + (block_size / ceil_feat_len) - 1) /
           (block_size / ceil_feat_len);
  block.y = ceil_feat_len / 32;
  block.x = (block_size + ceil_feat_len - 1) / ceil_feat_len * 32;
  selective_aggr_fwd_kernel<<<grid, block>>>(
      ptr.data<Index>(), idx.data<Index>(), input.data<float>(),
      output.data<float>(), mask.data<bool>(), num_node, feat_len);
}

void selective_aggr_bwd(Tensor grad_output, Tensor ptr, Tensor idx, Tensor mask,
                        Tensor computed_grad, Index num_center) {
  // computation
  int feat_len = grad_output.sizes().back();
  int block_size = 512;
  dim3 grid, block;
  int ceil_feat_len = ((feat_len + 31) / 32 * 32);
  block_size = std::max(block_size, ceil_feat_len);
  grid.x = (num_center + (block_size / ceil_feat_len) - 1) /
           (block_size / ceil_feat_len);
  block.y = ceil_feat_len / 32;
  block.x = (block_size + ceil_feat_len - 1) / ceil_feat_len * 32;
  selective_aggr_bwd_kernel<<<grid, block>>>(
      ptr.data<Index>(), idx.data<Index>(), grad_output.data<float>(),
      computed_grad.data<float>(), mask.data<bool>(), num_center, feat_len);
}

#define PAPER_NUM 121751666
#define PAPER_AUTHOR_NUM 244134778
#define PAPER_AUTHOR_INSTITUTE_NUM 244160499

__global__ void gen_edge_type_mag240m_kernel(Index *ptr, Index *idx,
                                             Index *sub_to_full, Index *etype,
                                             Index num_node) {
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  if (row >= num_node) return;
  Index begin = ptr[row], end = ptr[row + 1];
  Index center_id = sub_to_full[row];
#pragma unroll
  for (Index i = begin + lane; i < end; i += 32) {
    Index neighbor_id = sub_to_full[idx[i]];
    if (center_id < PAPER_NUM) {  // center=Paper
      if (neighbor_id < PAPER_NUM)
        etype[i] = 0;
      else if (neighbor_id < PAPER_AUTHOR_NUM)
        etype[i] = 1;
      else
        assert(0);                              // no paper-institute edge
    } else if (center_id < PAPER_AUTHOR_NUM) {  // center=Author
      if (neighbor_id < PAPER_NUM)
        etype[i] = 2;
      else if (neighbor_id < PAPER_AUTHOR_NUM)
        assert(0);  // no author-author edge
      else
        etype[i] = 3;
    } else {  // center=Institute
      if (neighbor_id < PAPER_NUM)
        assert(0);  // no institute-paper edge
      else if (neighbor_id < PAPER_AUTHOR_NUM)
        etype[i] = 4;
      else
        assert(0);  // no institute-institute edge
    }
  }
}

torch::Tensor gen_edge_type_mag240m(torch::Tensor ptr, torch::Tensor idx,
                                    torch::Tensor sub_to_full) {
  torch::Tensor etype = torch::zeros_like(idx);
  Index num_node = ptr.sizes()[0] - 1;
  int block_size = 512;
  int per_block = 512 / 32;
  gen_edge_type_mag240m_kernel<<<((num_node + per_block - 1) / per_block),
                                 block_size>>>(
      ptr.data<Index>(), idx.data<Index>(), sub_to_full.data<Index>(),
      etype.data<Index>(), num_node);
  return etype;
}