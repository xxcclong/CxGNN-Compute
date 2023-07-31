#include <queue>

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

std::vector<torch::Tensor> deg_schedule(Tensor csr_ptr, Tensor csr_idx,
                                        Tensor num_node_in_layer,
                                        int deg_thres) {
  std::vector<torch::Tensor> output;
  return output;
}

Tensor topo_sort(Tensor csr_ptr, Tensor csr_idx, Tensor degree) {
  Index num_node = degree.sizes()[0];
  Tensor output = torch::empty({num_node}, int64_option);
  Index *p_out = output.data<Index>();
  Index *p_ptr = csr_ptr.data<Index>();
  Index *p_idx = csr_idx.data<Index>();
  Index *p_deg = new Index[num_node];
  memcpy(p_deg, degree.data<Index>(), num_node * sizeof(Index));
  std::queue<Index> q;
  for (Index i = 0; i < num_node; ++i) {
    if (p_deg[i] == 0) {
      q.push(i);
    }
  }
  Index cnt = 0;
  while (cnt < num_node) {
    while (!q.empty()) {
      Index cur = q.front();
      q.pop();
      p_out[cnt++] = cur;
      for (Index i = p_ptr[cur]; i < p_ptr[cur + 1]; ++i) {
        Index next = p_idx[i];
        --p_deg[next];
        if (p_deg[next] == 0) {
          q.push(next);
        }
      }
    }
    // break;
    Index min_deg = num_node;
    Index to_in_queue = -1;
    for (Index i = 0; i < num_node; ++i) {
      if (p_deg[i] < min_deg && p_deg[i] > 1) {
        min_deg = p_deg[i];
        to_in_queue = i;
      } else if (p_deg[i] == 1) {
        p_deg[i] = 0;
        q.push(i);
      }
    }
    if (to_in_queue != -1) {
      q.push(to_in_queue);
      p_deg[to_in_queue] = 0;
    } else if (!q.empty()) {
      continue;
    } else {
      break;
    }
    if (cnt % 1000 == 0) std::cout << cnt << std::endl;
  }
  delete[] p_deg;
  if (cnt != num_node) {
    std::cout << "Error: graph is not a DAG, " << cnt << ' ' << num_node
              << std::endl;
  }
  return output;
}