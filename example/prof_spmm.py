import numpy as np
import torch

import cxgnncomp as cxgc
from cxgnncomp.codegen.util import compare


def prepare_data():
    torch.manual_seed(0)
    # dataset_name = "paper100m"
    # file_dir = "/home/huangkz/repos/new-diskgnn/DiskGNN/graph_loader/{}_batch.pt".format(
    #     dataset_name)
    file_dir = "/home/huangkz/repos/CxGNN-DL/dump.pt"
    batch = torch.load(file_dir)
    feat_len = 128
    x = torch.randn([batch["num_node_in_layer"][-1], feat_len],
                    dtype=torch.float32,
                    device='cuda')
    ptr = batch["ptr"].cuda()
    idx = batch["idx"].cuda()
    return x, ptr, idx, batch


def aggr_mm(x, ptr, idx, weight, num_center):
    # output = cxgc.sage_sum_forward(x, ptr, idx, num_center)
    output = cxgc.spmm_triton(x, ptr, idx, num_center)
    output = torch.mm(output, weight)
    return output


def mm_aggr(x, ptr, idx, weight, num_center):
    output = torch.mm(x, weight)
    output = cxgc.spmm_triton(output, ptr, idx, num_center)
    return output


def step_aggr_mm(x, ptr1, ptr2, idx1, idx2, weight, s1, s2, prep):
    with torch.cuda.stream(s1):
        output1 = cxgc.spmm_triton(x, ptr1, idx1, ptr1.shape[0] - 1)
        output1 = torch.mm(output1, weight)
    with torch.cuda.stream(s2):
        output2 = torch.mm(prep, weight)
        output2 = cxgc.spmm_triton(output2, ptr2, idx2, ptr2.shape[0] - 1)
    torch.cuda.synchronize()
    # output = torch.cat([output1, output2], dim=0)
    # return output


def start_from_zero(idx):
    mapping = {}
    cnt = 0
    unique_idx = []
    for item in idx:
        if not item in mapping:
            mapping[item] = cnt
            cnt += 1
            unique_idx.append(item)
    for i in range(len(idx)):
        idx[i] = mapping[idx[i]]
    return idx, unique_idx


def to_torch_cuda(arr):
    return torch.tensor(arr, dtype=torch.int64, device='cuda')


def preprocess(ptr, idx):
    ptr = ptr.cpu().numpy()
    idx = idx.cpu().numpy()
    node_group = []
    in_deg = []
    out_deg = np.zeros(max(idx) + 1, dtype=np.int32)
    thres = 20
    for i in idx:
        out_deg[i] += 1
    for i in range(ptr.shape[0] - 1):
        in_deg.append(ptr[i + 1] - ptr[i])

    new_ptr1 = []
    new_idx1 = []
    new_ptr2 = []
    new_idx2 = []
    new_ptr1.append(0)
    new_ptr2.append(0)
    target1 = []
    target2 = []
    for i in range(len(in_deg)):
        if in_deg[i] > thres:
            target1.append(i)
            new_ptr1.append(new_ptr1[-1] + in_deg[i])
            new_idx1.extend(idx[ptr[i]:ptr[i + 1]])
        else:
            target2.append(i)
            new_ptr2.append(new_ptr2[-1] + in_deg[i])
            new_idx2.extend(idx[ptr[i]:ptr[i + 1]])
    new_idx1, new_idx1_unique = start_from_zero(new_idx1)
    new_idx2, new_idx2_unique = start_from_zero(new_idx2)
    new_ptr1 = to_torch_cuda(new_ptr1)
    new_idx1 = to_torch_cuda(new_idx1)
    new_ptr2 = to_torch_cuda(new_ptr2)
    new_idx2 = to_torch_cuda(new_idx2)
    new_idx1_unique = to_torch_cuda(new_idx1_unique)
    new_idx2_unique = to_torch_cuda(new_idx2_unique)
    target1 = to_torch_cuda(target1)
    target2 = to_torch_cuda(target2)
    return new_ptr1, new_idx1, new_ptr2, new_idx2, new_idx1_unique, new_idx2_unique, target1, target2

    break_point = ptr.shape[0] // 2
    half_ptr1 = ptr[:break_point]
    half_idx1 = idx[:ptr[break_point]]
    half_ptr2 = ptr[break_point:] - ptr[break_point]
    half_idx2 = idx[ptr[break_point]:]
    cnt = 0
    mapping = {}
    cnt_mapping = {}
    mmax = 0
    for item in half_idx2:
        item = item.item()
        if item in mapping:
            cnt_mapping[item] += 1
            if cnt_mapping[item] > mmax:
                mmax = cnt_mapping[item]
        else:
            mapping[item] = cnt
            cnt_mapping[item] = 1
            cnt += 1
    print(cnt)
    print(half_idx2.shape)
    print(mmax)
    for i in range(half_idx2.shape[0]):
        half_idx2[i] = mapping[half_idx2[i].item()]
    half_idx2_unique = torch.unique(half_idx2)
    prep = torch.index_select(x, 0, half_idx2_unique)


if __name__ == "__main__":
    x, ptr, idx, b = prepare_data()
    print(b["num_node_in_layer"])
    # val = torch.randn(idx.shape, dtype=torch.float32, device='cuda')
    num_center = b["num_node_in_layer"][-2].item()
    output1 = cxgc.sage_sum_forward(x, ptr, idx, num_center)
    output2 = cxgc.spmm_triton(x, ptr, idx, num_center)
    compare(output1, output2)
    output = cxgc.prof("spmm_before", "triton",
                       lambda: cxgc.spmm_triton(x, ptr, idx, num_center))
    t_before_aggr = output[0]
    out_feat = 64
    weight = torch.randn([x.shape[1], out_feat],
                         dtype=torch.float32,
                         device='cuda')
    output = cxgc.prof("spmm", "manual",
                       lambda: cxgc.sage_sum_forward(x, ptr, idx, num_center))
    output = cxgc.prof("mm_before", "torch", lambda: torch.mm(x, weight))
    t_before_mm = output[0]
    output = cxgc.prof("mm_after", "torch", lambda: torch.mm(output1, weight))
    t_after_mm = output[0]

    after_mm = torch.mm(x, weight)
    output = cxgc.prof(
        "spmm_after", "triton",
        lambda: cxgc.spmm_triton(after_mm, ptr, idx, num_center))
    t_after_aggr = output[0]
    E = idx.shape[0]
    V_large = b["num_node_in_layer"][-1].item()
    V_small = b["num_node_in_layer"][-2].item()
    ratio1 = V_large / E
    ratio2 = t_before_mm / V_large
    ratio3 = (t_before_aggr - t_after_aggr) / E
    print(f"V_large / E {ratio1} MM {ratio2} AGGR {ratio3}")
    print(f"{ratio2 / (ratio1 * ratio2 - ratio3)}")

    # output1 = cxgc.sage_sum_forward_edge_value(x, ptr, idx, val, num_center)
    # output2 = cxgc.spmm_with_value_triton(x, ptr, idx, val, num_center)
    # compare(output1, output2)
    # cxgc.prof(
    #     "spmm_with_value", "triton",
    #     lambda: cxgc.spmm_with_value_triton(x, ptr, idx, val, num_center))
    # cxgc.prof(
    #     "spmm_with_value", "manual",
    #     lambda: cxgc.sage_sum_forward_edge_value(x, ptr, idx, val, num_center))

    weight_trans = weight.transpose(0, 1).contiguous()
    output1 = cxgc.spmm_mm_triton(x, ptr, idx, weight_trans, num_center)
    output2 = cxgc.sage_sum_forward(x, ptr, idx, num_center)
    output2 = torch.mm(output2, weight)
    # compare(output1, output2)
    # cxgc.prof("spmm_mm", "triton",
    #           lambda: cxgc.spmm_mm_triton(x, ptr, idx, weight, num_center))
    cxgc.prof("spmm_mm", "manual",
              lambda: aggr_mm(x, ptr, idx, weight, num_center))
    cxgc.prof("mm_spmm", "manual",
              lambda: mm_aggr(x, ptr, idx, weight, num_center))
    # output = torch.empty([num_center, out_feat], device='cuda')
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    new_ptr1, new_idx1, new_ptr2, new_idx2, new_idx1_unique, new_idx2_unique, target1, target2 = preprocess(
        ptr, idx)
    prep1 = torch.index_select(x, 0, new_idx1_unique)
    prep2 = torch.index_select(x, 0, new_idx2_unique)
    prep1_mm = torch.mm(prep1, weight)
    prep2_mm = torch.mm(prep2, weight)
    aggr_out1 = torch.randn([new_ptr1.shape[0] - 1, weight.shape[0]],
                            device='cuda')
    aggr_out2 = torch.randn([new_ptr2.shape[0] - 1, weight.shape[0]],
                            device='cuda')
    cxgc.prof(
        "half_spmm1", "triton", lambda: cxgc.spmm_triton(
            prep1, new_ptr1, new_idx1, new_ptr1.shape[0] - 1))
    cxgc.prof(
        "half_spmm2", "triton", lambda: cxgc.spmm_triton(
            prep2, new_ptr2, new_idx2, new_ptr2.shape[0] - 1))
    cxgc.prof("mm1", "torch", lambda: torch.mm(prep1, weight))
    cxgc.prof("mm2", "torch", lambda: torch.mm(prep2, weight))
    cxgc.prof(
        "half_spmm1_out", "triton", lambda: cxgc.spmm_triton(
            prep1_mm, new_ptr1, new_idx1, new_ptr1.shape[0] - 1))
    cxgc.prof(
        "half_spmm2_out", "triton", lambda: cxgc.spmm_triton(
            prep2_mm, new_ptr2, new_idx2, new_ptr2.shape[0] - 1))
    cxgc.prof("mm_out1", "torch", lambda: torch.mm(aggr_out1, weight))
    cxgc.prof("mm_out2", "torch", lambda: torch.mm(aggr_out2, weight))
    cxgc.prof(
        "step_spmm", "triton", lambda: step_aggr_mm(
            x, new_ptr1, new_ptr2, new_idx1, new_idx2, weight, s1, s2, prep2))

    print(new_ptr1.shape, new_idx1.shape, new_ptr2.shape, new_idx2.shape)