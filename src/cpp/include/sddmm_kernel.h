#pragma once

#include "common.h"


__global__ void sddmm_multihead(
    const Index *__restrict__ src, const Index *__restrict__ dst,
    const float *__restrict__ src_feat, const float *__restrict__ dst_feat,
    float *__restrict__ output, Index num_edge, int INFEATURE, int num_head);