#pragma once
#include <torch/extension.h>
#include <torch/torch.h>

#include "common.h"

using namespace torch::autograd;
using torch::Tensor;

enum SPMM_MULTIHEAD_SCHEDULE { Naive, PreReduce, Optimal };

torch::Tensor run_spmm_configurable(torch::Tensor ptr, torch::Tensor idx,
                                    torch::Tensor vin, Index num_node,
                                    int grid_x, int grid_y, int block_x,
                                    int block_y, int rpb, int cpb, int cpw,
                                    int grid_map, int block_map);

torch::Tensor run_spmm_configurable_int32(torch::Tensor ptr, torch::Tensor idx,
                                          torch::Tensor vin, Index num_node,
                                          int grid_x, int grid_y, int block_x,
                                          int block_y, int rpb, int cpb,
                                          int cpw, int grid_map, int block_map);

torch::Tensor spmm_multihead(torch::Tensor ptr, torch::Tensor idx,
                             torch::Tensor val, torch::Tensor vin,
                             Index num_node, SPMM_MULTIHEAD_SCHEDULE schedule,
                             int block_size);

void spmm_multihead_bwd(torch::Tensor ptr, torch::Tensor idx, torch::Tensor val,
                        torch::Tensor grad_output, torch::Tensor grad_x,
                        Index num_node, SPMM_MULTIHEAD_SCHEDULE schedule,
                        int block_size);

torch::Tensor run_spmm_multihead_configurable(
    torch::Tensor ptr, torch::Tensor idx, torch::Tensor val, torch::Tensor vin,
    Index num_node, int grid_x, int grid_y, int block_x, int block_y, int rpb,
    int cpb, int cpw, int grid_map, int block_map);