#pragma once
#include "common.h"

__global__ void run_spmm(const Index *__restrict__ ptr,
                         const Index *__restrict__ idx,
                         const float *__restrict__ vin,
                         float *__restrict__ vout, int num_node, int INFEATURE,
                         int rpb, int cpb, int cpw, int grid_map,
                         int block_map);

__global__ void run_spmm_sharedmem(const Index *__restrict__ ptr,
                                   const Index *__restrict__ idx,
                                   const float *__restrict__ vin,
                                   float *__restrict__ vout, int num_node,
                                   int INFEATURE, int rpb, int cpb, int cpw,
                                   int grid_map, int block_map);

__global__ void run_spmm_2(const Index *__restrict__ ptr,
                           const Index *__restrict__ idx,
                           const float *__restrict__ vin,
                           float *__restrict__ vout, int num_node,
                           int INFEATURE, int rpb, int cpb, int cpw,
                           int grid_map, int block_map);

__global__ void run_spmm_4(const Index *__restrict__ ptr,
                           const Index *__restrict__ idx,
                           const float *__restrict__ vin,
                           float *__restrict__ vout, int num_node,
                           int INFEATURE, int rpb, int cpb, int cpw,
                           int grid_map, int block_map);

__global__ void run_spmm_8(const Index *__restrict__ ptr,
                           const Index *__restrict__ idx,
                           const float *__restrict__ vin,
                           float *__restrict__ vout, int num_node,
                           int INFEATURE, int rpb, int cpb, int cpw,
                           int grid_map, int block_map);

__global__ void run_spmm_sharedmem_2(const Index *__restrict__ ptr,
                                     const Index *__restrict__ idx,
                                     const float *__restrict__ vin,
                                     float *__restrict__ vout, int num_node,
                                     int INFEATURE, int rpb, int cpb, int cpw,
                                     int grid_map, int block_map);

__global__ void run_spmm_sharedmem_4(const Index *__restrict__ ptr,
                                     const Index *__restrict__ idx,
                                     const float *__restrict__ vin,
                                     float *__restrict__ vout, int num_node,
                                     int INFEATURE, int rpb, int cpb, int cpw,
                                     int grid_map, int block_map);

__global__ void run_spmm_sharedmem_8(const Index *__restrict__ ptr,
                                     const Index *__restrict__ idx,
                                     const float *__restrict__ vin,
                                     float *__restrict__ vout, int num_node,
                                     int INFEATURE, int rpb, int cpb, int cpw,
                                     int grid_map, int block_map);

__global__ void run_spmm_sharedmem_step_2(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ vin, float *__restrict__ vout, int num_node,
    int INFEATURE, int rpb, int cpb, int cpw, int grid_map, int block_map);

__global__ void run_spmm_sharedmem_step_4(
    const Index *__restrict__ ptr, const Index *__restrict__ idx,
    const float *__restrict__ vin, float *__restrict__ vout, int num_node,
    int INFEATURE, int rpb, int cpb, int cpw, int grid_map, int block_map);

__global__ void run_spmm_sharedmem_int(const int *__restrict__ ptr,
                                       const int *__restrict__ idx,
                                       const float *__restrict__ vin,
                                       float *__restrict__ vout, int num_node,
                                       int INFEATURE, int rpb, int cpb, int cpw,
                                       int grid_map, int block_map);

__global__ void run_spmm_sharedmem_step_2_int(
    const int *__restrict__ ptr, const int *__restrict__ idx,
    const float *__restrict__ vin, float *__restrict__ vout, int num_node,
    int INFEATURE, int rpb, int cpb, int cpw, int grid_map, int block_map);

__global__ void run_spmm_sharedmem_step_4_int(
    const int *__restrict__ ptr, const int *__restrict__ idx,
    const float *__restrict__ vin, float *__restrict__ vout, int num_node,
    int INFEATURE, int rpb, int cpb, int cpw, int grid_map, int block_map);

__global__ void run_spmm_sharedmem_step_8_int(
    const int *__restrict__ ptr, const int *__restrict__ idx,
    const float *__restrict__ vin, float *__restrict__ vout, int num_node,
    int INFEATURE, int rpb, int cpb, int cpw, int grid_map, int block_map);