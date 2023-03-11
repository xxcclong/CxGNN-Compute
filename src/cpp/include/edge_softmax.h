#pragma once
#include <torch/extension.h>
#include <torch/torch.h>

#include "common.h"
using namespace torch::autograd;

class EdgeSoftMaxFunction : public Function<EdgeSoftMaxFunction> {
 public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor ptr,
                               torch::Tensor idx,
                               torch::Tensor att_dst /*[num_dst,num_head]*/,
                               torch::Tensor att_src /*[num_src, num_head]*/,
                               Index num_edge, float relu_l);
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

// performing edge attention for GAT
// output = softmax(leaky_relu(att_dst + att_src))
torch::Tensor edge_softmax_forward(
    torch::Tensor ptr, torch::Tensor idx,
    torch::Tensor att_dst /*[num_dst,num_head]*/,
    torch::Tensor att_src /*[num_src, num_head]*/, Index num_edge,
    float relu_l);

torch::Tensor edge_value_degree(torch::Tensor ptr, Index num_dst,
                                Index num_edge);

torch::Tensor spmv(torch::Tensor ptr, torch::Tensor idx, torch::Tensor input,
                   Index num_dst);