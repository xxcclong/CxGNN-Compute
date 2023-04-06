import torch
import numpy


def neighbor_grouping(ptr, neighbor_thres=32):
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


def reorder_by(ptr, idx, metric):
    print("reorder_by: not reordering source nodes")
    assert metric.shape[0] == ptr.shape[0] - 1
    ptr = ptr.cpu().numpy()
    num_src = torch.max(idx) + 1
    idx = idx.cpu().numpy()
    # new_indices = torch.empty([num_src])
    # new_indices[metric] = torch.arange(metric.shape[0], device=metric.device)
    new_ptr = [0]
    new_idx = []
    for i in range(len(ptr) - 1):
        deg = ptr[metric[i] + 1] - ptr[metric[i]]
        new_ptr.append(new_ptr[-1] + deg)
        for j in range(ptr[metric[i]], ptr[metric[i] + 1]):
            # new_idx.append(new_indices[idx[j]])
            new_idx.append(idx[j])
    return torch.from_numpy(numpy.array(new_ptr)), torch.from_numpy(
        numpy.array(new_idx))
