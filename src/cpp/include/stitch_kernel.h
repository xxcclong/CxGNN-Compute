#pragma once

#include <torch/extension.h>
#include <torch/torch.h>

#include "common.h"

using namespace torch::autograd;
using torch::Tensor;

void pad_rel_gpu(Tensor rel, Tensor idx, Tensor count, int thres, int num_rel,
                 Index base);