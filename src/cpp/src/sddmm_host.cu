#include "sddmm_host.h"
#include "sddmm_kernel.h"

void run_sddmm(Tensor src, Tensor dst, Tensor src_feat, Tensor dst_feat,
               Tensor output, Index num_edge) {
  int num_head = src_feat.sizes()[1];
  dim3 grid, block;
  block.x = 256;
  grid.x = ceil_div(num_edge * num_head, block.x / 32);
  sddmm_multihead<<<grid, block>>>(
      src.data_ptr<Index>(), dst.data_ptr<Index>(), src_feat.data_ptr<float>(),
      dst_feat.data_ptr<float>(), output.data_ptr<float>(), num_edge,
      src_feat.sizes().back(), num_head);
}

Tensor run_sddmm_vertex_centric(Tensor ptr, Tensor idx, Tensor src_feat,
                                Tensor dst_feat, int num_node, int grid_x,
                                int grid_y, int block_x, int block_y, int rpb,
                                int cpb, int cpw, int grid_map, int block_map) {
  int num_head = src_feat.sizes()[1];  // [num_source_ode, num_head, feat_size]
  int feat_len = src_feat.sizes().back();
  Index num_edge = idx.sizes()[0];
  auto output = dst_feat.new_zeros({num_edge, num_head});
  if (cpw == 32) {
    sddmm_multihead_vertex_centric_1<<<dim3(grid_x, grid_y, 1),
                                       dim3(block_x, block_y, 1),
                                       block_x * block_y * sizeof(int)>>>(
        ptr.data_ptr<Index>(), idx.data_ptr<Index>(),
        src_feat.data_ptr<float>(), dst_feat.data_ptr<float>(),
        output.data_ptr<float>(), num_node, feat_len, num_head, rpb, cpb, cpw,
        grid_map, block_map);
  } else if (cpw == 64) {
    sddmm_multihead_vertex_centric_2<<<dim3(grid_x, grid_y, 1),
                                       dim3(block_x, block_y, 1),
                                       block_x * block_y * sizeof(int)>>>(
        ptr.data_ptr<Index>(), idx.data_ptr<Index>(),
        src_feat.data_ptr<float>(), dst_feat.data_ptr<float>(),
        output.data_ptr<float>(), num_node, feat_len, num_head, rpb, cpb, cpw,
        grid_map, block_map);
  } else if (cpw == 128) {
    sddmm_multihead_vertex_centric_4<<<dim3(grid_x, grid_y, 1),
                                       dim3(block_x, block_y, 1),
                                       block_x * block_y * sizeof(int)>>>(
        ptr.data_ptr<Index>(), idx.data_ptr<Index>(),
        src_feat.data_ptr<float>(), dst_feat.data_ptr<float>(),
        output.data_ptr<float>(), num_node, feat_len, num_head, rpb, cpb, cpw,
        grid_map, block_map);
  } else if (cpw == 256) {
    sddmm_multihead_vertex_centric_8<<<dim3(grid_x, grid_y, 1),
                                       dim3(block_x, block_y, 1),
                                       block_x * block_y * sizeof(int)>>>(
        ptr.data_ptr<Index>(), idx.data_ptr<Index>(),
        src_feat.data_ptr<float>(), dst_feat.data_ptr<float>(),
        output.data_ptr<float>(), num_node, feat_len, num_head, rpb, cpb, cpw,
        grid_map, block_map);
  } else {
    ASSERTWITH(false, "cpw must be 32, 64, 128 or 256");
  }
  return output;
}