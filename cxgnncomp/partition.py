import torch


def partition_2d_gpu(edge_index, num_device, rank_dst, rank_src, device=None):
    if device == None:
        device = edge_index.device
    num_node = torch.max(edge_index) + 1
    num_node_per_device = num_node // num_device
    # print(edge_index.shape)
    lower_bound_dst = num_node_per_device * rank_dst
    upper_bound_dst = num_node_per_device * (
        rank_dst + 1) if rank_dst < num_device - 1 else num_node
    lower_bound_src = num_node_per_device * rank_src
    upper_bound_src = num_node_per_device * (
        rank_src + 1) if rank_src < num_device - 1 else num_node
    edge_index = edge_index[:,
                            torch.logical_and(
                                torch.
                                logical_and(edge_index[1] >= lower_bound_dst,
                                            edge_index[1] < upper_bound_dst),
                                torch.
                                logical_and(edge_index[0] >= lower_bound_src,
                                            edge_index[0] < upper_bound_src))]
    # print(edge_index.shape)
    degree = torch.bincount(edge_index[1],
                            minlength=num_node).to(device)
    target_ids = torch.arange(num_node,
                              dtype=torch.int64,
                              device=device)
    assert degree.shape[0] == num_node
    new_degree = degree[degree > 0]
    target_ids = target_ids[degree > 0] - lower_bound_dst
    degree = new_degree
    new_ptr = torch.cumsum(degree, dim=0).to(torch.int64)
    new_ptr = torch.cat(
        [torch.tensor([0], dtype=torch.int64, device=new_ptr.device), new_ptr])
    _, indices = torch.sort(edge_index[1], descending=True)
    new_idx = edge_index[0][indices] - lower_bound_src
    return new_ptr, new_idx, target_ids