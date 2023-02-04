#pragma once
#include <torch/extension.h>
#include <torch/torch.h>

#include "common.h"


using namespace torch::autograd;
using torch::Tensor;

void run_sddmm(Tensor src, Tensor dst, Tensor src_feat, Tensor dst_feat,
               Tensor output, Index num_edge);