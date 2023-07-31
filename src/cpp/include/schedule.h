#include <torch/extension.h>
#include <torch/torch.h>

#include <vector>

#include "common.h"

using torch::Tensor;

std::vector<torch::Tensor> rel_schedule(Tensor csr_ptr, Tensor csr_idx,
                                        Tensor rel, Tensor num_node_in_layer,
                                        int num_rel);

std::vector<torch::Tensor> deg_schedule(Tensor csr_ptr, Tensor csr_idx,
                                        Tensor num_node_in_layer,
                                        int deg_thres);

Tensor topo_sort(Tensor csr_ptr, Tensor csr_idx, Tensor degree);