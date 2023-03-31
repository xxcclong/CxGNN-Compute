#pragma once

#include "common.h"

__global__ void sddmm_multihead(const Index *__restrict__ src,
                                const Index *__restrict__ dst,
                                const float *__restrict__ src_feat,
                                const float *__restrict__ dst_feat,
                                float *__restrict__ output, Index num_edge,
                                int INFEATURE, int num_head);

__global__ void sddmm_multihead_vertex_centric_1(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ src_feat, const float *__restrict__ dst_feat,
    float *__restrict__ output, int num_node, int INFEATURE, int num_head,
    int rpb, int cpb, int cpw, int grid_map, int block_map);

__global__ void sddmm_multihead_vertex_centric_2(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ src_feat, const float *__restrict__ dst_feat,
    float *__restrict__ output, int num_node, int INFEATURE, int num_head,
    int rpb, int cpb, int cpw, int grid_map, int block_map);

__global__ void sddmm_multihead_vertex_centric_4(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ src_feat, const float *__restrict__ dst_feat,
    float *__restrict__ output, int num_node, int INFEATURE, int num_head,
    int rpb, int cpb, int cpw, int grid_map, int block_map);

__global__ void sddmm_multihead_vertex_centric_8(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ src_feat, const float *__restrict__ dst_feat,
    float *__restrict__ output, int num_node, int INFEATURE, int num_head,
    int rpb, int cpb, int cpw, int grid_map, int block_map);