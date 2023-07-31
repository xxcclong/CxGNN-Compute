#pragma once

#include <torch/extension.h>
#include <torch/torch.h>

#include "common.h"

using namespace torch::autograd;
using torch::Tensor;

class AggrRelFunction : public Function<AggrRelFunction> {
 public:
  static Tensor forward(AutogradContext *ctx, Tensor input, Tensor ptr,
                        Tensor idx, Tensor etype, Index num_node, int num_rel);
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

class AggrRelDirectFunction : public Function<AggrRelDirectFunction> {
 public:
  static Tensor forward(AutogradContext *ctx, Tensor input, Tensor ptr,
                        Tensor idx, Tensor weights, Tensor etype,
                        Index num_node, int num_rel);
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

Tensor aggr_rel(Tensor input, Tensor ptr, Tensor idx, Tensor etype,
                Index num_node, int num_rel);
Tensor aggr_rel_direct(Tensor input, Tensor ptr, Tensor idx, Tensor weights,
                       Tensor etype, Index num_node, int num_rel);

torch::Tensor aggr_rgcn_direct_func(torch::Tensor input, torch::Tensor ptr,
                                    torch::Tensor idx, torch::Tensor weights,
                                    torch::Tensor rel, Index num_node);

void run_typed_linear(Tensor vin, Tensor weights, Tensor output, Tensor types,
                      int in_feat_tile);

void run_typed_linear_s2e(Tensor vin, Tensor weights, Tensor output,
                          Tensor src_id, Tensor types, int in_feat_tile);

void run_typed_linear_s2d(Tensor vin, Tensor weights, Tensor output,
                          Tensor src_id, Tensor dst_id, Tensor types,
                          int in_feat_tile);