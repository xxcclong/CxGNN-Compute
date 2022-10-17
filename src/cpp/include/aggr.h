#ifndef AGGR_H
#define AGGR_H
#include <torch/torch.h>

#include "common.h"

using namespace torch::autograd;

enum class AggrType { Mean, Sum };

class SAGEFunction : public Function<SAGEFunction> {
 public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input,
                               torch::Tensor ptr, torch::Tensor idx,
                               int num_node, AggrType aggr_type);
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

class SAGEEdgeValueFunction : public Function<SAGEEdgeValueFunction> {
 public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input,
                               torch::Tensor ptr, torch::Tensor idx,
                               torch::Tensor edge_value, int num_node,
                               AggrType aggr_type);
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

class GatherFunction : public Function<GatherFunction> {
 public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input,
                               torch::Tensor dest, Index num_node);
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

class AggrRelFunction : public Function<AggrRelFunction> {
 public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input,
                               torch::Tensor ptr, torch::Tensor idx,
                               torch::Tensor etype, Index num_node,
                               int num_rel);
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

class AggrRelDirectFunction : public Function<AggrRelDirectFunction> {
 public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input,
                               torch::Tensor ptr, torch::Tensor idx,
                               torch::Tensor weights, torch::Tensor etype,
                               Index num_node, int num_rel);
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

// class AggrWithTargetFunction : public Function<AggrWithTargetFunction> {
// public:
//   static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input,
//   torch::Tensor ptr,
//                                torch::Tensor idx, torch::Tensor target, Index
//                                num_node);
//   static tensor_list backward(AutogradContext *ctx, tensor_list
//   grad_outputs);
// };

torch::Tensor sage_sum_forward_edge_value(torch::Tensor input,
                                          torch::Tensor ptr, torch::Tensor idx,
                                          torch::Tensor edge_value,
                                          int num_node);
torch::Tensor sage_mean_forward_edge_value(torch::Tensor input,
                                           torch::Tensor ptr, torch::Tensor idx,
                                           torch::Tensor edge_value,
                                           int num_node);
torch::Tensor sage_sum_forward(torch::Tensor input, torch::Tensor ptr,
                               torch::Tensor idx, int num_node);
torch::Tensor sage_mean_forward(torch::Tensor input, torch::Tensor ptr,
                                torch::Tensor idx, int num_node);
torch::Tensor gather(torch::Tensor input /* O(E) */,
                     torch::Tensor dest /* O(E) */, Index num_node);
torch::Tensor aggr_rel(torch::Tensor input, torch::Tensor ptr,
                       torch::Tensor idx, torch::Tensor etype, Index num_node,
                       int num_rel);
torch::Tensor aggr_rel_direct(torch::Tensor input, torch::Tensor ptr,
                              torch::Tensor idx, torch::Tensor weights,
                              torch::Tensor etype, Index num_node, int num_rel);

torch::Tensor get_graph_structure_score(torch::Tensor ptr, torch::Tensor idx,
                                        Index num_node, Index num_seed,
                                        int num_layer);

// torch::Tensor sage_sum_target_forward(torch::Tensor input, torch::Tensor ptr,
// torch::Tensor idx,
//                                       torch::Tensor target, int num_node);
#endif