#pragma once
#include <torch/extension.h>
#include <torch/torch.h>

#include "common.h"

using namespace torch::autograd;
using torch::Tensor;

void run_sddmm(Tensor src, Tensor dst, Tensor src_feat, Tensor dst_feat,
               Tensor output, Index num_edge);

Tensor run_sddmm_vertex_centric(Tensor ptr, Tensor idx, Tensor src_feat,
                                Tensor dst_feat, int num_node, int grid_x,
                                int grid_y, int block_x, int block_y, int rpb,
                                int cpb, int cpw, int grid_map, int block_map);