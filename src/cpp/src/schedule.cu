#include "schedule.h"

std::vector<torch::Tensor> rel_schedule(Tensor csr_ptr, Tensor csr_idx,
                                        Tensor rel, Tensor num_node_in_layer,
                                        int num_rel) {
  std::vector<torch::Tensor> output;
  int num_layer = num_node_in_layer.sizes()[0] - 1;
  auto rel_num_node_in_layer_tensor =
      torch::empty({num_rel, num_layer}, int64_option);
  // num_rel * num_layer
  Index *p_num = rel_num_node_in_layer_tensor.data<Index>();
  //   printf(
  //       "num_layer: %d, num_rel: %d, num_node_in_layer: %d
  //       rel_num_node_in_layer "
  //       "%d\n",
  //       num_layer, num_rel, num_node_in_layer.sizes().size(),
  //       rel_num_node_in_layer_tensor.sizes().size());
  for (int rel_iter = 0; rel_iter < num_rel; ++rel_iter) {
    int layer_id = 0;
    std::vector<Index> sub_ptr;
    sub_ptr.push_back(0);
    std::vector<Index> sub_idx;
    std::vector<Index> sub_target;
    auto ptr_vec = csr_ptr.data<Index>();
    auto idx_vec = csr_idx.data<Index>();
    auto rel_vec = rel.data<int>();
    auto num_node_in_layer_vec = num_node_in_layer.data<Index>();
    for (Index i = 0; i < csr_ptr.sizes()[0] - 1; ++i) {
      Index cnt = 0;
      for (Index j = ptr_vec[i]; j < ptr_vec[i + 1]; ++j) {
        if (rel_iter == rel_vec[j]) {
          ++cnt;
          sub_idx.push_back(idx_vec[j]);
        }
      }
      if (cnt > 0) {
        sub_target.push_back(i);
        sub_ptr.push_back(sub_ptr.back() + cnt);
      }
      if (i == num_node_in_layer_vec[layer_id] - 1) {
        p_num[layer_id + num_layer * rel_iter] = sub_target.size();
        ++layer_id;
      }
    }

    auto ptr_tensor = torch::empty({(int)sub_ptr.size()}, int64_option);
    memcpy(ptr_tensor.data<Index>(), sub_ptr.data(),
           sub_ptr.size() * sizeof(Index));
    auto idx_tensor = torch::empty({(int)sub_idx.size()}, int64_option);
    memcpy(idx_tensor.data<Index>(), sub_idx.data(),
           sub_idx.size() * sizeof(Index));
    auto target_tensor = torch::empty({(int)sub_target.size()}, int64_option);
    memcpy(target_tensor.data<Index>(), sub_target.data(),
           sub_target.size() * sizeof(Index));
    output.push_back(ptr_tensor);
    output.push_back(idx_tensor);
    output.push_back(target_tensor);
  }
  output.push_back(rel_num_node_in_layer_tensor);
  return output;
}