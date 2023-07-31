#pragma once

#include "common.h"

__global__ void spmm_multihead_pre_reduce(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ val, const float *__restrict__ vin,
    float *__restrict__ vout, int num_node, int INFEATURE, int num_head);

__global__ void spmm_multihead_sharedmem_1(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ val, const float *__restrict__ vin,
    float *__restrict__ vout, int num_node, int INFEATURE, int num_head,
    int rpb, int cpb, int cpw, int grid_map, int block_map);

__global__ void spmm_multihead_sharedmem_2(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ val, const float *__restrict__ vin,
    float *__restrict__ vout, int num_node, int INFEATURE, int num_head,
    int rpb, int cpb, int cpw, int grid_map, int block_map);

__global__ void spmm_multihead_sharedmem_4(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ val, const float *__restrict__ vin,
    float *__restrict__ vout, int num_node, int INFEATURE, int num_head,
    int rpb, int cpb, int cpw, int grid_map, int block_map);

__global__ void spmm_multihead_sharedmem_8(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ val, const float *__restrict__ vin,
    float *__restrict__ vout, int num_node, int INFEATURE, int num_head,
    int rpb, int cpb, int cpw, int grid_map, int block_map);