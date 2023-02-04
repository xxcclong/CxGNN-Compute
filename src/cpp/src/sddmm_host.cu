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