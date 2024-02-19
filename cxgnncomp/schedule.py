import torch
import numpy
import time


def neighbor_grouping_gpu(ptr, neighbor_thres=32):
    deg = ptr[1:] - ptr[:-1]
    up_deg = (deg + neighbor_thres - 1) // neighbor_thres
    total_num = torch.sum(up_deg)
    new_ptr = torch.ones(total_num + 1, dtype=torch.int64,
                         device=ptr.device) * neighbor_thres
    new_ptr[0] = 0
    incorrect_pos = torch.cumsum(up_deg, dim=0)
    new_ptr[incorrect_pos] = deg - neighbor_thres * (up_deg - 1)
    new_ptr = torch.cumsum(new_ptr, dim=0)
    # new_target = torch.arange(num_node, dtype=torch.int64, device=ptr.device).repeat_interleave(up_deg)
    return new_ptr


def neighbor_grouping(ptr, neighbor_thres=32):
    if not isinstance(ptr, numpy.ndarray):
        ptr = ptr.cpu().numpy()
    new_ptr = [0]
    new_target = []
    for i in range(len(ptr) - 1):
        end = ptr[i + 1]
        start = ptr[i]
        while end - start > neighbor_thres:
            start += neighbor_thres
            new_ptr.append(start)
            new_target.append(i)
        new_ptr.append(end)
        new_target.append(i)
    return torch.from_numpy(numpy.array(new_ptr)).cuda(), torch.from_numpy(
        numpy.array(new_target)).cuda()


def partition_2d(ptr, idx, num_src, num_dst, num_parts=8):
    num_src = int(num_src)
    num_dst = int(num_dst)
    ptr = ptr.cpu().numpy()
    idx = idx.cpu().numpy()
    new_ptr = [0]
    new_idx = []
    new_target = []
    num_src_per_part = num_src // num_parts
    num_dst_per_part = num_dst // num_parts
    for i in range(num_parts):
        for j in range(num_parts):
            for k in range(len(ptr) - 1):
                if k < num_dst_per_part * i or (k >= num_dst_per_part * (i + 1)
                                                and i != num_parts - 1):
                    continue
                end = ptr[k + 1]
                start = ptr[k]
                cnt = 0
                for m in range(start, end):
                    if idx[m] >= num_src_per_part * j and (
                            idx[m] < num_src_per_part *
                        (j + 1) or j == num_parts - 1):
                        new_idx.append(idx[m])
                        new_target.append(k)
                        cnt += 1
                if cnt > 0:
                    new_ptr.append(new_ptr[-1] + cnt)
    assert len(new_idx) == idx.shape[0]
    assert len(new_idx) == new_ptr[-1]
    return torch.from_numpy(numpy.array(new_ptr)).cuda(), torch.from_numpy(
        numpy.array(new_idx)).cuda()


'''
Assuming the dst and src are not representing the same set of nodes
e.g.:
dst: 0 0 0 1 1 2 2 2 2
src: a b c d e f g h i
metric: 2 0 1
new_dst: 2 2 2 2 0 0 0 1 1
new_src: f g h i a b c d e 
'''


def reorder_by(ptr, idx, metric):
    t0 = time.time()
    device = ptr.device
    print("reorder_by: not reordering source nodes")
    assert metric.shape[0] == ptr.shape[0] - 1
    ptr = ptr.cpu().numpy()
    num_src = torch.max(idx) + 1
    idx = idx.cpu().numpy()
    metric = metric.cpu().numpy()
    # new_indices = torch.empty([num_src])
    # new_indices[metric] = torch.arange(metric.shape[0], device=metric.device)
    new_ptr = [0]
    new_idx = []
    for i in range(len(ptr) - 1):
        curr_node_id = metric[i]
        deg = ptr[curr_node_id + 1] - ptr[curr_node_id]
        new_ptr.append(new_ptr[-1] + deg)
        new_idx += idx[ptr[curr_node_id]:ptr[curr_node_id + 1]].tolist()
    print("reorder_by: time elapsed: ", time.time() - t0)
    return torch.from_numpy(numpy.array(new_ptr)).to(device), torch.from_numpy(
        numpy.array(new_idx)).to(device)


def remove_from_graph(ptr, idx, remove_flag):
    t0 = time.time()
    device = ptr.device
    ptr = ptr.cpu().numpy()
    idx = idx.cpu().numpy()
    remove_flag = remove_flag.cpu().numpy()
    new_ptr = [0]
    new_idx = []
    for i in range(len(ptr) - 1):
        if not remove_flag[i]:
            deg = ptr[i + 1] - ptr[i]
            new_ptr.append(new_ptr[-1] + deg)
            new_idx += idx[ptr[i]:ptr[i + 1]].tolist()
    print("remove: time elapsed: ", time.time() - t0)
    return torch.from_numpy(numpy.array(new_ptr)).to(device), torch.from_numpy(
        numpy.array(new_idx)).to(device)


def remove_from_graph_by_edge(ptr, idx, remove_flag):
    assert remove_flag.shape == idx.shape, f"remove_flag.shape != idx.shape {remove_flag.shape} {idx.shape}"
    t0 = time.time()
    device = ptr.device
    ptr = ptr.cpu().numpy()
    idx = idx.cpu().numpy()
    remove_flag = remove_flag.cpu().numpy()
    new_ptr = [0]
    new_idx = []
    for i in range(len(ptr) - 1):
        deg = 0
        for j in range(ptr[i], ptr[i + 1]):
            if not remove_flag[j]:
                new_idx.append(idx[j])
                deg += 1
        if deg > 0:
            new_ptr.append(new_ptr[-1] + deg)
    print("remove: time elapsed: ", time.time() - t0)
    print("after removing", len(new_ptr), len(new_idx))
    return torch.from_numpy(numpy.array(new_ptr)).to(device), torch.from_numpy(
        numpy.array(new_idx)).to(device)