#include "edge_softmax.h"

__global__ void edge_softmax_fwd(Index *ptr, Index *idx, float *att_dst,
                                 float *att_src, bool *output_mid,
                                 float *newval, Index num_dst, int num_head,
                                 float relu_l, bool *used_mask = nullptr) {
  int dst_id =
      ((blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5)) / num_head;
  int head_id =
      ((blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5)) % num_head;
  int laneid = threadIdx.x & 31;
  if (dst_id < num_dst && (used_mask == nullptr || used_mask[dst_id])) {
    float dst = att_dst[dst_id * num_head + head_id];
    Index begin = ptr[dst_id], end = ptr[dst_id + 1];
    float max_val = -100000.f;
    for (Index i = begin + laneid; i < end; i += 32) {
      float tmpsum = dst + att_src[idx[i] * num_head + head_id];  // dst + src
      tmpsum = max(tmpsum, tmpsum * relu_l);                      // leaky relu
      newval[i * num_head + head_id] = tmpsum;
      output_mid[i * num_head + head_id] = tmpsum > 0;
      max_val = max(max_val, tmpsum);  // max
    }
    for (int i = 16; i > 0; i >>= 1) {
      max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, i));
    }
    max_val = __shfl_sync(0xffffffff, max_val, 0);
    float partial_sum = 0.f;
    for (int i = begin + laneid; i < end; i += 32) {
      // exp(edgevalue - max)
      float tmp = __expf(newval[i * num_head + head_id] - max_val);
      newval[i * num_head + head_id] = tmp;
      partial_sum += tmp;
    }
    for (int i = 16; i > 0; i >>= 1) {
      partial_sum += __shfl_down_sync(0xffffffff, partial_sum, i);  // sum
    }
    partial_sum = __shfl_sync(0xffffffff, partial_sum, 0);
    for (int i = begin + laneid; i < end; i += 32) {
      newval[i * num_head + head_id] /= partial_sum;  // /= sum
    }
  }
}

__global__ void edge_softmax_bwd(Index *ptr, Index *idx,
                                 float *grad /*[E, num_head]*/,
                                 bool *output_mid /*[E, num_head]*/,
                                 float *newval /*[E, num_head]*/,
                                 float *output_grad_dst /*[num_dst, num_head]*/,
                                 float *output_grad_src /*[num_src, num_head]*/,
                                 Index num_dst, int num_head, float relu_l,
                                 bool *used_mask = nullptr) {
  int dst_id =
      ((blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5)) / num_head;
  int head_id =
      ((blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5)) % num_head;
  int laneid = threadIdx.x & 31;
  if (dst_id < num_dst && (used_mask == nullptr || used_mask[dst_id])) {
    float partial_sum = 0.f;
    Index begin = ptr[dst_id], end = ptr[dst_id + 1];
    for (int i = begin + laneid; i < end; i += 32) {
      partial_sum +=
          newval[i * num_head + head_id] * grad[i * num_head + head_id];
    }
    for (int i = 16; i > 0; i >>= 1) {
      partial_sum += __shfl_down_sync(0xffffffff, partial_sum, i);  // sum
    }
    partial_sum =
        __shfl_sync(0xffffffff, partial_sum, 0);  // reduce(out * grad)
    float dst_grad = 0.f;
    int pos_flag = 0;
    int neg_flag = 0;
    for (int i = begin + laneid; i < end; i += 32) {
      float tmp = newval[i * num_head + head_id] *
                  (grad[i * num_head + head_id] - partial_sum);
      if (!output_mid[i * num_head + head_id]) {
        tmp *= relu_l;
        neg_flag = 1;
      } else {
        pos_flag = 1;
      }
      dst_grad += tmp;
      // atomicAdd(output_grad_dst + dst_id * num_head + head_id, tmp);
      atomicAdd(output_grad_src + idx[i] * num_head + head_id, tmp);
    }

    for (int i = 16; i > 0; i >>= 1) {
      pos_flag += __shfl_down_sync(0xffffffff, pos_flag, i);  // sum
      neg_flag += __shfl_down_sync(0xffffffff, neg_flag, i);  // sum
      dst_grad += __shfl_down_sync(0xffffffff, dst_grad, i);  // sum
    }
    if (laneid == 0 && pos_flag > 0 && neg_flag > 0) {
      output_grad_dst[dst_id * num_head + head_id] = dst_grad;
    }
  }
}

torch::Tensor edgeAttentionForwardImpl(
    AutogradContext *ctx, torch::Tensor ptr, torch::Tensor idx,
    torch::Tensor att_dst /*[num_dst,num_head]*/,
    torch::Tensor att_src /*[num_src, num_head]*/, Index num_edge,
    float relu_l) {
  ctx->saved_data["ptr"] = (int64_t)ptr.data<Index>();
  ctx->saved_data["idx"] = (int64_t)idx.data<Index>();
  int num_dst = att_dst.sizes()[0];
  int num_src = att_src.sizes()[0];
  int num_head = att_dst.sizes()[1];
  ctx->saved_data["num_dst"] = (int64_t)num_dst;
  ctx->saved_data["num_src"] = (int64_t)num_src;
  ctx->saved_data["num_edge"] = (int64_t)num_edge;
  ctx->saved_data["num_head"] = (int64_t)num_head;
  ctx->saved_data["relu_l"] = (double)relu_l;

  bool *used_mask = nullptr;

  ASSERT(att_dst.device().index() >= 0);
  checkCudaErrors(cudaSetDevice(att_dst.device().index()));
  auto output_mid =
      att_dst.new_empty({num_edge, num_head}, torch::dtype(torch::kBool));
  auto output = att_dst.new_empty({num_edge, num_head});
  // auto output = torch::zeros_like(input);
  output.requires_grad_(true);
  int block_size = 256;
  dim3 grid, block;
  grid.x = (num_dst * num_head + (block_size / 32) - 1) / (block_size / 32);
  block.x = block_size;
  edge_softmax_fwd<<<grid, block>>>(
      ptr.data<Index>(), idx.data<Index>(), att_dst.data<float>(),
      att_src.data<float>(), output_mid.data<bool>(), output.data<float>(),
      num_dst, num_head, relu_l, used_mask);
  ctx->save_for_backward({output, output_mid});
  return output;
}

torch::Tensor EdgeSoftMaxFunction::forward(
    AutogradContext *ctx, torch::Tensor ptr, torch::Tensor idx,
    torch::Tensor att_dst /*[num_dst,num_head]*/,
    torch::Tensor att_src /*[num_src, num_head]*/, Index num_edge,
    float relu_l) {
  return edgeAttentionForwardImpl(ctx, ptr, idx, att_dst, att_src, num_edge,
                                  relu_l);
}

tensor_list edgeAttentionBackwardImpl(AutogradContext *ctx,
                                      tensor_list grad_outputs) {
  auto saved = ctx->get_saved_variables();
  auto newval = saved[0];
  auto output_mid = saved[1];
  Index *ptr = (Index *)(ctx->saved_data["ptr"].toInt());
  Index *idx = (Index *)(ctx->saved_data["idx"].toInt());
  float relu_l = (float)(ctx->saved_data["relu_l"].toDouble());
  auto grad_output = grad_outputs[0];

  int num_dst = ctx->saved_data["num_dst"].toInt();
  int num_src = ctx->saved_data["num_src"].toInt();
  int num_edge = ctx->saved_data["num_edge"].toInt();
  int num_head = ctx->saved_data["num_head"].toInt();
  auto grad_att_dst = grad_output.new_zeros({num_dst, num_head});
  auto grad_att_src = grad_output.new_zeros({num_src, num_head});
  int block_size = 256;
  bool *used_mask = nullptr;

  dim3 grid, block;
  grid.x = (num_dst * num_head + (block_size / 32) - 1) / (block_size / 32);
  block.x = block_size;
  edge_softmax_bwd<<<grid, block>>>(
      ptr, idx, grad_output.data<float>(), output_mid.data<bool>(),
      newval.data<float>(), grad_att_dst.data<float>(),
      grad_att_src.data<float>(), num_dst, num_head, relu_l, used_mask);
  return {grad_att_dst, grad_att_src};
}

tensor_list EdgeSoftMaxFunction::backward(AutogradContext *ctx,
                                          tensor_list grad_outputs) {
  auto outputs = edgeAttentionBackwardImpl(ctx, grad_outputs);
  return {torch::Tensor(), torch::Tensor(), outputs[0],
          outputs[1],      torch::Tensor(), torch::Tensor()};
}

torch::Tensor edge_softmax_forward(torch::Tensor ptr, torch::Tensor idx,
                                   torch::Tensor att_dst, torch::Tensor att_src,
                                   Index num_edge, float relu_l) {
  return EdgeSoftMaxFunction::apply(ptr, idx, att_dst, att_src, num_edge,
                                    relu_l);
}

__global__ void gen_edge_value_degree(Index *ptr, float *output,
                                      Index num_dst) {
  int dst_id = ((blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5));
  int laneid = threadIdx.x & 31;
  if (dst_id < num_dst) {
    Index begin = ptr[dst_id], end = ptr[dst_id + 1];
    assert(begin < end);
    float val = 1.f / (end - begin);
    for (int i = begin + laneid; i < end; i += 32) {
      output[i] = val;
    }
  }
}

torch::Tensor edge_value_degree(torch::Tensor ptr, Index num_dst,
                                Index num_edge) {
  torch::TensorOptions option = torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .device(ptr.device())
                                    .requires_grad(false);
  auto output = torch::empty({num_edge}, option);
  int block_size = 256;
  dim3 grid, block;
  grid.x = (num_dst + (block_size / 32) - 1) / (block_size / 32);
  block.x = block_size;
  gen_edge_value_degree<<<grid, block>>>(ptr.data<Index>(),
                                         output.data<float>(), num_dst);
  return output;
}