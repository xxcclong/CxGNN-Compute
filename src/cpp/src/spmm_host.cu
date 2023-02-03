#include "aggr_kernel.h"
#include "spmm_host.h"
#include "spmm_kernel.h"
#include "spmm_multihead_kernel.h"

torch::Tensor run_spmm_configurable(torch::Tensor ptr, torch::Tensor idx,
                                    torch::Tensor vin, Index num_node,
                                    int grid_x, int grid_y, int block_x,
                                    int block_y, int rpb, int cpb, int cpw,
                                    int grid_map, int block_map) {
  ASSERTWITH(vin.dim() == 2, "vin must be 2D");
  int feat_len = vin.sizes().back();
  auto output = vin.new_zeros({num_node, feat_len});
  if (cpw == 32) {
    run_spmm_sharedmem<<<dim3(grid_x, grid_y, 1), dim3(block_x, block_y, 1),
                         block_x * block_y * sizeof(int)>>>(
        ptr.data<Index>(), idx.data<Index>(), vin.data<float>(),
        output.data<float>(), num_node, feat_len, rpb, cpb, cpw, grid_map,
        block_map);
    // run_spmm<<<dim3(grid_x, grid_y, 1), dim3(block_x, block_y, 1)>>>(
    //     ptr.data<Index>(), idx.data<Index>(), vin.data<float>(),
    //     output.data<float>(), num_node, feat_len, rpb, cpb, cpw, grid_map,
    //     block_map);
  } else if (cpw == 64) {
    run_spmm_sharedmem_2<<<dim3(grid_x, grid_y, 1), dim3(block_x, block_y, 1),
                           block_x * block_y * sizeof(int)>>>(
        ptr.data<Index>(), idx.data<Index>(), vin.data<float>(),
        output.data<float>(), num_node, feat_len, rpb, cpb, cpw, grid_map,
        block_map);
    // run_spmm_2<<<dim3(grid_x, grid_y, 1), dim3(block_x, block_y, 1)>>>(
    //     ptr.data<Index>(), idx.data<Index>(), vin.data<float>(),
    //     output.data<float>(), num_node, feat_len, rpb, cpb, cpw, grid_map,
    //     block_map);

  } else if (cpw == 128) {
    run_spmm_sharedmem_4<<<dim3(grid_x, grid_y, 1), dim3(block_x, block_y, 1),
                           block_x * block_y * sizeof(int)>>>(
        ptr.data<Index>(), idx.data<Index>(), vin.data<float>(),
        output.data<float>(), num_node, feat_len, rpb, cpb, cpw, grid_map,
        block_map);
    // run_spmm_4<<<dim3(grid_x, grid_y, 1), dim3(block_x, block_y, 1)>>>(
    //     ptr.data<Index>(), idx.data<Index>(), vin.data<float>(),
    //     output.data<float>(), num_node, feat_len, rpb, cpb, cpw, grid_map,
    //     block_map);
  } else if (cpw == 256) {
    run_spmm_sharedmem_8<<<dim3(grid_x, grid_y, 1),
                           dim3(block_x, block_y, 1)>>>(
        ptr.data<Index>(), idx.data<Index>(), vin.data<float>(),
        output.data<float>(), num_node, feat_len, rpb, cpb, cpw, grid_map,
        block_map);
  } else {
    ASSERT(0);
  }
  return output;
}

torch::Tensor run_spmm_configurable_int32(torch::Tensor ptr, torch::Tensor idx,
                                          torch::Tensor vin, Index num_node,
                                          int grid_x, int grid_y, int block_x,
                                          int block_y, int rpb, int cpb,
                                          int cpw, int grid_map,
                                          int block_map) {
  ASSERTWITH(vin.dim() == 2, "vin must be 2D");
  int feat_len = vin.sizes().back();
  auto output = vin.new_zeros({num_node, feat_len});
  if (cpw == 32) {
    run_spmm_sharedmem_int<<<dim3(grid_x, grid_y, 1), dim3(block_x, block_y, 1),
                             block_x * block_y * sizeof(int)>>>(
        ptr.data<int>(), idx.data<int>(), vin.data<float>(),
        output.data<float>(), num_node, feat_len, rpb, cpb, cpw, grid_map,
        block_map);
    // run_spmm<<<dim3(grid_x, grid_y, 1), dim3(block_x, block_y, 1)>>>(
    //     ptr.data<Index>(), idx.data<Index>(), vin.data<float>(),
    //     output.data<float>(), num_node, feat_len, rpb, cpb, cpw, grid_map,
    //     block_map);
  } else if (cpw == 64) {
    run_spmm_sharedmem_step_2_int<<<dim3(grid_x, grid_y, 1),
                                    dim3(block_x, block_y, 1),
                                    block_x * block_y * sizeof(int)>>>(
        ptr.data<int>(), idx.data<int>(), vin.data<float>(),
        output.data<float>(), num_node, feat_len, rpb, cpb, cpw, grid_map,
        block_map);
    // run_spmm_2<<<dim3(grid_x, grid_y, 1), dim3(block_x, block_y, 1)>>>(
    //     ptr.data<Index>(), idx.data<Index>(), vin.data<float>(),
    //     output.data<float>(), num_node, feat_len, rpb, cpb, cpw, grid_map,
    //     block_map);

  } else if (cpw == 128) {
    run_spmm_sharedmem_step_4_int<<<dim3(grid_x, grid_y, 1),
                                    dim3(block_x, block_y, 1),
                                    block_x * block_y * sizeof(int)>>>(
        ptr.data<int>(), idx.data<int>(), vin.data<float>(),
        output.data<float>(), num_node, feat_len, rpb, cpb, cpw, grid_map,
        block_map);
    // run_spmm_4<<<dim3(grid_x, grid_y, 1), dim3(block_x, block_y, 1)>>>(
    //     ptr.data<Index>(), idx.data<Index>(), vin.data<float>(),
    //     output.data<float>(), num_node, feat_len, rpb, cpb, cpw, grid_map,
    //     block_map);
  } else if (cpw == 256) {
    run_spmm_sharedmem_step_8_int<<<dim3(grid_x, grid_y, 1),
                                    dim3(block_x, block_y, 1),
                                    block_x * block_y * sizeof(int)>>>(
        ptr.data<int>(), idx.data<int>(), vin.data<float>(),
        output.data<float>(), num_node, feat_len, rpb, cpb, cpw, grid_map,
        block_map);
    // run_spmm_8<<<dim3(grid_x, grid_y, 1), dim3(block_x, block_y, 1)>>>(
    //     ptr.data<Index>(), idx.data<Index>(), vin.data<float>(),
    //     output.data<float>(), num_node, feat_len, rpb, cpb, cpw, grid_map,
    //     block_map);
  } else {
    ASSERT(0);
  }
  return output;
}

torch::Tensor spmm_multihead(torch::Tensor ptr, torch::Tensor idx,
                             torch::Tensor val, torch::Tensor vin,
                             Index num_node, SPMM_MULTIHEAD_SCHEDULE schedule,
                             int block_size) {
  ASSERTWITH(vin.dim() == 3, "vin must be 3D, [num_node, num_head, feat_len]");
  ASSERTWITH(val.dim() == 2, "val must be 2D, [num_edge, num_head]");
  int feat_len = vin.sizes().back();
  int num_head = vin.sizes()[1];
  int ceil_feat_len = ceil(feat_len, 32);
  ASSERTWITH(num_head == val.sizes()[1], "num_head mismatch {} {}", num_head,
             val.sizes()[1]);
  dim3 grid, block;
  if (block_size < ceil_feat_len) block_size = ceil_feat_len;
  block.y = ceil_feat_len / 32;
  block.x = ceil_div(block_size, ceil_feat_len) * 32;
  switch (schedule) {
    case SPMM_MULTIHEAD_SCHEDULE::Naive: {
      grid.x = ceil_div(num_node * num_head, block_size / ceil_feat_len);
      auto output = vin.new_zeros({num_node, num_head, feat_len});
      gen_fwd_sum_edge_value_multi_head<<<grid, block>>>(
          ptr.data<Index>(), idx.data<Index>(), val.data<float>(),
          vin.data<float>(), output.data<float>(), num_node, feat_len,
          num_head);
      return output;
    }
    case SPMM_MULTIHEAD_SCHEDULE::PreReduce:
    case SPMM_MULTIHEAD_SCHEDULE::Optimal: {
      grid.x = ceil_div(num_node, block_size / ceil_feat_len);
      auto output = vin.new_zeros({num_node, feat_len});
      spmm_multihead_pre_reduce<<<grid, block>>>(
          ptr.data<Index>(), idx.data<Index>(), val.data<float>(),
          vin.data<float>(), output.data<float>(), num_node, feat_len,
          num_head);
      return output;
    }
    default:
      ASSERT(0);
  }
}

torch::Tensor run_spmm_multihead_configurable(
    torch::Tensor ptr, torch::Tensor idx, torch::Tensor val, torch::Tensor vin,
    Index num_node, int grid_x, int grid_y, int block_x, int block_y, int rpb,
    int cpb, int cpw, int grid_map, int block_map) {
  ASSERTWITH(vin.dim() == 3, "vin must be 2D");
  int feat_len = vin.sizes().back();
  int num_head = vin.sizes()[1];
  auto output = vin.new_zeros({num_node, feat_len});
  if (cpw != 128) return output;
  if (cpw == 32) {
    ASSERT(0);
  } else if (cpw == 64) {
    ASSERT(0);
  } else if (cpw == 128) {
    spmm_multihead_sharedmem_4<<<dim3(grid_x, grid_y, 1),
                                 dim3(block_x, block_y, 1),
                                 block_x * block_y * sizeof(int)>>>(
        ptr.data<Index>(), idx.data<Index>(), val.data<float>(),
        vin.data<float>(), output.data<float>(), num_node, feat_len, num_head,
        rpb, cpb, cpw, grid_map, block_map);
  } else if (cpw == 256) {
    ASSERT(0);
  } else {
    ASSERT(0);
  }
  return output;
}