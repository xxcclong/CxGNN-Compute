#ifndef AGGR_H
#define AGGR_H
#include <torch/extension.h>
#include <torch/torch.h>

#include "common.h"

using namespace torch::autograd;
using torch::Tensor;

enum class AggrType { Mean, Sum };

class SAGEFunction : public Function<SAGEFunction> {
 public:
  static Tensor forward(AutogradContext *ctx, Tensor input, Tensor ptr,
                        Tensor idx, int num_node, AggrType aggr_type);
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

class SAGEEdgeValueFunction : public Function<SAGEEdgeValueFunction> {
 public:
  static Tensor forward(AutogradContext *ctx, Tensor input, Tensor ptr,
                        Tensor idx, Tensor edge_value, int num_node,
                        AggrType aggr_type);
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

class GatherFunction : public Function<GatherFunction> {
 public:
  static Tensor forward(AutogradContext *ctx, Tensor input, Tensor dest,
                        Index num_node);
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

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

class SelectiveAggrFunction : public Function<SelectiveAggrFunction> {
 public:
  static Tensor forward(AutogradContext *ctx, Tensor input, Tensor ptr,
                        Tensor idx, Tensor mask, Tensor output, int num_node);
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

// class AggrWithTargetFunction : public Function<AggrWithTargetFunction> {
// public:
//   static Tensor forward(AutogradContext *ctx, Tensor input,
//   Tensor ptr,
//                                Tensor idx, Tensor target, Index
//                                num_node);
//   static tensor_list backward(AutogradContext *ctx, tensor_list
//   grad_outputs);
// };

Tensor sage_sum_forward_edge_value(Tensor input, Tensor ptr, Tensor idx,
                                   Tensor edge_value, int num_node);
Tensor sage_mean_forward_edge_value(Tensor input, Tensor ptr, Tensor idx,
                                    Tensor edge_value, int num_node);
Tensor sage_sum_forward(Tensor input, Tensor ptr, Tensor idx, int num_node);
Tensor sage_mean_forward(Tensor input, Tensor ptr, Tensor idx, int num_node);
Tensor gather(Tensor input /* O(E) */, Tensor dest /* O(E) */, Index num_node);
Tensor aggr_rel(Tensor input, Tensor ptr, Tensor idx, Tensor etype,
                Index num_node, int num_rel);
Tensor aggr_rel_direct(Tensor input, Tensor ptr, Tensor idx, Tensor weights,
                       Tensor etype, Index num_node, int num_rel);

void selective_aggr_fwd(Tensor input, Tensor ptr, Tensor idx, Tensor mask,
                        Tensor output, int num_node);
void selective_aggr_bwd(Tensor grad_output, Tensor ptr, Tensor idx, Tensor mask,
                        Tensor computed_grad, Index num_center);

void target_sage_sum_forward(Tensor input, Tensor ptr, Tensor idx,
                             Tensor targets, Tensor output, int num_node);

torch::Tensor aggr_rgcn_direct_func(torch::Tensor input, torch::Tensor ptr,
                                    torch::Tensor idx, torch::Tensor weights,
                                    torch::Tensor rel, Index num_node);

// Tensor sage_sum_target_forward(Tensor input, Tensor ptr,
// Tensor idx,
//                                       Tensor target, int num_node);

torch::Tensor gen_edge_type_mag240m(torch::Tensor ptr, torch::Tensor idx,
                                    torch::Tensor sub_to_full);

torch::Tensor run_spmm_configurable(torch::Tensor ptr, torch::Tensor idx,
                                    torch::Tensor vin, Index num_node,
                                    int grid_x, int grid_y, int block_x,
                                    int block_y, int rpb, int cpb, int cpw,
                                    int grid_map, int block_map);
#endif
