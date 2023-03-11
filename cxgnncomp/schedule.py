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