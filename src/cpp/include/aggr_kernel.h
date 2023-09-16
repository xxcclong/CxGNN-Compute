#pragma once
#include "common.h"

__global__ void gen_fwd_mean(Index *ptr, Index *idx, float *vin, float *vout,
                             int num_node, int INFEATURE);

__global__ void gen_fwd_sum(Index *ptr, Index *idx, float *vin, float *vout,
                            int num_node, int INFEATURE);

__global__ void gen_fwd_mean_edge_value(Index *ptr, Index *idx,
                                        float *edge_value, float *vin,
                                        float *vout, int num_node,
                                        int INFEATURE);

__global__ void gen_fwd_sum_edge_value(Index *ptr, Index *idx,
                                       float *edge_value, float *vin,
                                       float *vout, int num_node,
                                       int INFEATURE);

__global__ void gen_fwd_mean_edge_value_multi_head(Index *ptr, Index *idx,
                                                   float *edge_value,
                                                   float *vin, float *vout,
                                                   int num_node, int INFEATURE,
                                                   int num_head);

__global__ void gen_fwd_sum_edge_value_multi_head(Index *ptr, Index *idx,
                                                  float *edge_value, float *vin,
                                                  float *vout, int num_node,
                                                  int INFEATURE, int num_head);

__global__ void gen_bwd_mean(
    Index *ptr, Index *idx, float *grads_in, float *vout_fwd, float *grads_out,
    int num_node,
    int INFEATURE);  // push the gradient to the neighbor vertex

__global__ void gen_bwd_sum(
    Index *ptr, Index *idx, float *grads_in, float *vout_fwd, float *grads_out,
    int num_node,
    int INFEATURE);  // push the gradient to the neighbor vertex

__global__ void gen_bwd_mean_edge_value(
    Index *ptr, Index *idx, float *edge_value, float *grads_in, float *vout_fwd,
    float *grads_out, int num_node,
    int INFEATURE);  // push the gradient to the neighbor vertex

__global__ void gen_bwd_sum_edge_value(
    Index *ptr, Index *idx, float *edge_value, float *grads_in, float *vout_fwd,
    float *grads_out, int num_node,
    int INFEATURE);  // push the gradient to the neighbor vertex

__global__ void gen_bwd_sum_edge_value_edge_grad(
    Index *ptr, Index *idx, float *edge_value, float *grads_in, float *vout_fwd,
    float *grads_out, int num_node, int INFEATURE, float *edge_grad);

__global__ void gen_bwd_mean_edge_value_multi_head(
    Index *ptr, Index *idx, float *edge_value, float *grads_in, float *vout_fwd,
    float *grads_out, int num_node, int INFEATURE,
    int num_head);  // push the gradient to the neighbor vertex

__global__ void gen_bwd_sum_edge_value_multi_head(
    Index *ptr, Index *idx, float *edge_value, float *grads_in, float *vout_fwd,
    float *grads_out, int num_node, int INFEATURE,
    int num_head);  // push the gradient to the neighbor vertex

__global__ void gen_bwd_sum_edge_value_multi_head_edge_grad(
    Index *ptr, Index *idx, float *edge_value, float *grads_in, float *vout_fwd,
    float *grads_out, int num_node, int INFEATURE, int num_head,
    float *edge_grad);

__global__ void selective_aggr_fwd_kernel(Index *ptr, Index *idx, float *vin,
                                          float *vout, bool *mask, int num_node,
                                          int INFEATURE);

__global__ void selective_aggr_bwd_kernel(Index *ptr, Index *idx,
                                          float *grads_in, float *grads_out,
                                          bool *mask, Index num_node,
                                          int INFEATURE);

__global__ void target_aggr(Index *ptr, Index *idx, Index *targets, float *vin,
                            float *vout, int num_node, int INFEATURE);


__global__ void target_aggr_backward(Index *ptr, Index *idx, Index *targets, const float *grads_in,
                            float *grads_out, int num_node, int INFEATURE);