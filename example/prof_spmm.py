import numpy as np
import torch
import cxgnncomp as cxgc
import cxgnncomp_backend
from cxgnncomp.codegen.util import compare
from torch_scatter import segment_csr, gather_csr


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


def sort_graph(ptr, idx):
    # ptr = ptr.cpu().numpy()
    # idx = idx.cpu().numpy()
    ptr = np.array(ptr)
    idx = np.array(idx)
    deg = torch.from_numpy(ptr[1:] - ptr[:-1])
    sorted, indices = torch.sort(deg, descending=True)
    new_ptr = []
    new_idx = []
    new_ptr.append(0)
    for i in indices:
        new_ptr.append(new_ptr[-1] + deg[i])
        new_idx.extend(idx[ptr[i]:ptr[i + 1]])
    return new_ptr, new_idx


def partition_node_dim(ptr, idx, target):
    num_parts = 8
    num_element_per_part = (len(ptr) - 1) // num_parts + 1
    new_ptrs = []
    new_idxs = []
    new_targets = []
    # ptr = ptr.cpu().numpy()
    # idx = idx.cpu().numpy()
    for i in range(num_parts):
        if i == num_parts - 1:
            new_ptrs.append(ptr[i * num_element_per_part:] -
                            ptr[i * num_element_per_part])
            new_idxs.append(idx[ptr[i * num_element_per_part]:ptr[-1]])
            new_targets.append(target[i * num_element_per_part:])
        else:
            new_ptrs.append(ptr[i * num_element_per_part:(i + 1) *
                                num_element_per_part] -
                            ptr[i * num_element_per_part])
            new_idxs.append(
                idx[ptr[i * num_element_per_part]:ptr[(i + 1) *
                                                      num_element_per_part]])
            new_targets.append(target[i * num_element_per_part:(i + 1) *
                                      num_element_per_part])
    return new_ptrs, new_idxs, new_targets


def preprocess_neighbor_partition(ptr, idx, num_node_overall):
    num_parts = 8
    num_element_per_part = (num_node_overall // num_parts) + 1
    new_ptrs = []
    new_idxs = []
    new_targets = []
    for _ in range(num_parts):
        new_ptrs.append([0])
        new_idxs.append([])
        new_targets.append([])
    ptr = ptr.cpu().numpy()
    idx = idx.cpu().numpy()
    for i in range(len(ptr) - 1):
        cnts = [0 for _ in range(num_parts)]
        for j in range(ptr[i], ptr[i + 1]):
            k = idx[j] // num_element_per_part
            # print(k, len(new_idxs[k]), len(idx), k)
            new_idxs[k].append(idx[j])
            cnts[k] += 1
        for k, cnt in enumerate(cnts):
            if cnt > 0:
                new_ptrs[k].append(new_ptrs[k][-1] + cnt)
                new_targets[k].append(i)
    for i in range(num_parts):
        new_ptrs[i] = to_torch_cuda(new_ptrs[i])
        # new_idxs[i], _ = start_from_zero(new_idxs[i])
        new_idxs[i] = to_torch_cuda(new_idxs[i])
        new_targets[i] = to_torch_cuda(new_targets[i])
    node_dim = True
    if node_dim:
        ret_ptrs = []
        ret_idxs = []
        ret_targets = []
        ret_ptrs.append(new_ptrs[0])
        ret_idxs.append(new_idxs[0])
        ret_targets.append(new_targets[0])
        for i in range(1, num_parts):  # not the first part
            retp, reti, rett = partition_node_dim(new_ptrs[i], new_idxs[i],
                                                  new_targets[i])
            ret_ptrs.extend(retp)
            ret_idxs.extend(reti)
            ret_targets.extend(rett)
        return ret_ptrs, ret_idxs, ret_targets
    else:
        return new_ptrs, new_idxs, new_targets


def get_sub_mat(ptr, idx, node_l, node_h, neighbor_l, neighbor_h):
    ptr = ptr.cpu().numpy()
    idx = idx.cpu().numpy()
    new_ptr = [0]
    new_idx = []
    for i in range(len(ptr) - 1):
        if i < node_l or i >= node_h:
            continue
        cnt = 0
        for j in range(ptr[i], ptr[i + 1]):
            if idx[j] >= neighbor_l and idx[j] < neighbor_h:
                cnt += 1
                new_idx.append(idx[j])
        if cnt > 0:
            new_ptr.append(new_ptr[-1] + cnt)
    return to_torch_cuda(new_ptr), to_torch_cuda(new_idx)


# partition the graph by degree
def preprocess(ptr, idx):
    thres = 32
    to_sort = False
    ptr = ptr.cpu().numpy()
    idx = idx.cpu().numpy()
    node_group = []
    # in_deg = []
    # out_deg = np.zeros(max(idx) + 1, dtype=np.int32)
    # for i in idx:
    #     out_deg[i] += 1
    # for i in range(ptr.shape[0] - 1):
    #     in_deg.append(ptr[i + 1] - ptr[i])
    in_deg = ptr[1:] - ptr[:-1]

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
    if to_sort:
        new_ptr1, new_idx1 = sort_graph(new_ptr1, new_idx1)
        new_ptr2, new_idx2 = sort_graph(new_ptr2, new_idx2)
    new_ptr1 = to_torch_cuda(new_ptr1)
    new_idx1 = to_torch_cuda(new_idx1)
    new_ptr2 = to_torch_cuda(new_ptr2)
    new_idx2 = to_torch_cuda(new_idx2)
    new_idx1_unique = to_torch_cuda(new_idx1_unique)
    new_idx2_unique = to_torch_cuda(new_idx2_unique)
    target1 = to_torch_cuda(target1)
    target2 = to_torch_cuda(target2)
    print("======== Partitioned Graph ========")
    print(new_ptr1.shape, new_ptr2.shape, ptr.shape)
    print(new_idx1.shape, new_idx2.shape, idx.shape)
    print(new_idx1_unique.shape, new_idx2_unique.shape)
    print("======== Finish Partitioned Graph ========")
    return new_ptr1, new_idx1, new_ptr2, new_idx2, new_idx1_unique, new_idx2_unique, target1, target2


best_configs = {
    "papers100M": (91096, 1, 256, 2, 16, 128, 128, 1, 1),
    "arxiv": None,
}


def test_partition_by_degree():
    overall_ans = {}
    # x, ptr, idx, b = cxgc.prepare_data_sampled_graph(dset="papers100M",
    #                                                  num_seeds=10000)
    # x, ptr, idx, b = cxgc.prepare_data()
    dset = "products"
    x, ptr, idx, b = cxgc.prepare_data_full_graph(dset)
    output = cxgc.tune_spmm(ptr.shape[0] - 1, idx.shape[0], x.shape[1],
                            cxgnncomp_backend.run_spmm_configurable,
                            [ptr, idx, x, ptr.shape[0] - 1])
    best_configs[dset] = output[1]
    output = cxgc.tune_spmm(ptr.shape[0] - 1, idx.shape[0], x.shape[1],
                            cxgnncomp_backend.run_spmm_configurable,
                            [ptr, idx, x, ptr.shape[0] - 1],
                            best_configs[dset])
    # output = cxgc.tune_spmm(x, ptr, idx, best_configs[dset])
    print("tuned spmm in", output[0], output[0] / idx.shape[0])
    overall_ans["tune spmm before"] = output[0]
    # x, ptr, idx, b = prepare_data_full_graph()
    # preprocess(ptr, idx)
    # exit()
    print(b["num_node_in_layer"])
    # val = torch.randn(idx.shape, dtype=torch.float32, device='cuda')
    num_center = b["num_node_in_layer"][-2].item()
    output1 = cxgc.sage_sum_forward(x, ptr, idx, num_center)
    output2 = cxgc.spmm_triton(x, ptr, idx, num_center)
    compare(output1, output2)
    output = cxgc.prof("spmm_before", "triton",
                       lambda: cxgc.spmm_triton(x, ptr, idx, num_center))
    t_before_aggr = output[0]
    print("spmm in", output[0], output[0] / idx.shape[0])
    overall_ans["triton spmm before"] = output[0]
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
    print("spmm out", output[0], output[0] / idx.shape[0])
    overall_ans["triton spmm after"] = output[0]
    output = cxgc.tune_spmm(after_mm, ptr, idx)
    print("tune spmm out", output[0], output[0] / idx.shape[0])
    overall_ans["tune spmm after"] = output[0]
    t_after_aggr = output[0]
    # E = idx.shape[0]
    # V_large = b["num_node_in_layer"][-1].item()
    # V_small = b["num_node_in_layer"][-2].item()
    # ratio1 = V_large / E
    # ratio2 = t_before_mm / V_large
    # ratio3 = (t_before_aggr - t_after_aggr) / E
    # print(f"V_large / E {ratio1} MM {ratio2} AGGR {ratio3}")
    # print(f"{ratio2 / (ratio1 * ratio2 - ratio3)}")

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

    # Aggregation
    output = cxgc.prof(
        "half_spmm1", "triton", lambda: cxgc.spmm_triton(
            prep1, new_ptr1, new_idx1, new_ptr1.shape[0] - 1))
    print("half_spmm1", output[0], output[0] / new_idx1.shape[0])
    overall_ans["triton spmm before 1"] = output[0]
    output = cxgc.prof(
        "half_spmm2", "triton", lambda: cxgc.spmm_triton(
            prep2, new_ptr2, new_idx2, new_ptr2.shape[0] - 1))
    print("half_spmm2", output[0], output[0] / new_idx2.shape[0])
    overall_ans["triton spmm before 2"] = output[0]

    # Tune
    output = cxgc.tune_spmm(prep1, new_ptr1, new_idx1, None)
    print("tune in1", output[0], output[0] / new_idx1.shape[0])
    overall_ans["tune spmm before 1"] = output[0]
    output = cxgc.tune_spmm(prep2, new_ptr2, new_idx2, None)
    print("tune in2", output[0], output[0] / new_idx2.shape[0])
    overall_ans["tune spmm before 2"] = output[0]

    # Matmul
    output = cxgc.prof("mm1", "torch", lambda: torch.mm(prep1, weight))
    print("mm1 flop",
          prep1.shape[0] * prep1.shape[1] * weight.shape[1] / 1e9 / output[0])
    output = cxgc.prof("mm2", "torch", lambda: torch.mm(prep2, weight))
    print("mm2 flop",
          prep2.shape[0] * prep2.shape[1] * weight.shape[1] / 1e9 / output[0])

    # Aggregate on matmuled tensor
    output = cxgc.prof(
        "half_spmm1_out", "triton", lambda: cxgc.spmm_triton(
            prep1_mm, new_ptr1, new_idx1, new_ptr1.shape[0] - 1))
    overall_ans["triton spmm after 1"] = output[0]
    output = cxgc.prof(
        "half_spmm2_out", "triton", lambda: cxgc.spmm_triton(
            prep2_mm, new_ptr2, new_idx2, new_ptr2.shape[0] - 1))
    overall_ans["triton spmm after 2"] = output[0]

    # Tune on matmuled tensor
    output = (cxgc.tune_spmm(prep1_mm, new_ptr1, new_idx1, None))
    print("tune out1", output[0], output[0] / new_idx1.shape[0])
    overall_ans["tune spmm after 1"] = output[0]
    output = (cxgc.tune_spmm(prep2_mm, new_ptr2, new_idx2, None))
    print("tune out2", output[0], output[0] / new_idx2.shape[0])
    overall_ans["tune spmm after 2"] = output[0]

    # matmul on aggregated tensor
    output = cxgc.prof("mm_out1", "torch", lambda: torch.mm(aggr_out1, weight))
    print(
        "mmout1 flop", aggr_out1.shape[0] * aggr_out1.shape[1] *
        weight.shape[1] / 1e9 / output[0])
    output = cxgc.prof("mm_out2", "torch", lambda: torch.mm(aggr_out2, weight))
    print(
        "mmout2 flop", aggr_out2.shape[0] * aggr_out2.shape[1] *
        weight.shape[1] / 1e9 / output[0])

    # Stepped SpMM
    cxgc.prof(
        "step_spmm", "triton", lambda: step_aggr_mm(
            x, new_ptr1, new_ptr2, new_idx1, new_idx2, weight, s1, s2, prep2))

    print(new_ptr1.shape, new_idx1.shape, new_ptr2.shape, new_idx2.shape)
    for k, v in overall_ans.items():
        print(k, v)


def spmm_by_op(x, ptr, idx):
    expanded = torch.index_select(x, 0, idx)
    output_op = segment_csr(expanded, ptr, reduce="sum")
    return output_op


def test_partition_by_neighbor():
    # x, ptr, idx, b = cxgc.prepare_data_sampled_graph(dset="papers100M",
    #                                                  num_seeds=10000)
    # x, ptr, idx, b = cxgc.prepare_data()
    dset = "products"
    x, ptr, idx, b = cxgc.prepare_data_full_graph(dset)
    print("full graph", ptr.shape, idx.shape)
    new_ptrs, new_idxs, new_targets = preprocess_neighbor_partition(
        ptr, idx, b["num_node_in_layer"][-1].item())
    time_total = 0
    for i in range(len(new_ptrs)):
        print(f"subgraph {i}", new_ptrs[i].shape, new_idxs[i].shape,
              new_targets[i].shape, new_ptrs[i][-1])
        # if i == 0:
        if 1:
            output = cxgc.tune_spmm(x, new_ptrs[i], new_idxs[i])
        else:
            output = cxgc.prof(f"spmm {i}", "op",
                               lambda: spmm_by_op(x, new_ptrs[i], new_idxs[i]))
        print(output, output[0] / new_idxs[i].shape[0])
        time_total += output[0]
    print("total time", time_total)

    output = cxgc.tune_spmm(x, ptr, idx)
    print(output)


def test_single_submat():
    dset = "arxiv"
    x, ptr, idx, b = cxgc.prepare_data_full_graph(dset)
    print("full graph", ptr.shape, idx.shape)
    num_node = b["num_node_in_layer"][-1].item()
    num_center = ptr.shape[0] - 1
    num_parts = 8
    # sparse large
    new_ptr, new_idx = get_sub_mat(ptr, idx, num_center // num_parts,
                                   num_center, num_node // num_parts, num_node)
    print(new_ptr.shape, new_idx.shape, new_ptr[-1])
    # output = cxgc.tune_spmm(x, new_ptr, new_idx)

    output = cxgc.tune_spmm(ptr.shape[0] - 1, idx.shape[0], x.shape[1],
                            cxgnncomp_backend.run_spmm_configurable,
                            [ptr, idx, x, ptr.shape[0] - 1])
    print(output)
    new_ptr, new_idx = get_sub_mat(ptr, idx, 0, num_center // num_parts, 0,
                                   num_node // num_parts)
    print(new_ptr.shape, new_idx.shape, new_ptr[-1])
    output = cxgc.tune_spmm(x, new_ptr, new_idx)

    new_ptr, new_idx = get_sub_mat(ptr, idx, 0, num_center // num_parts,
                                   num_node // num_parts, num_node)
    print(new_ptr.shape, new_idx.shape, new_ptr[-1])
    output = cxgc.tune_spmm(x, new_ptr, new_idx)

    new_ptr, new_idx = get_sub_mat(ptr, idx, num_center // num_parts,
                                   num_center, 0, num_node // num_parts)
    print(new_ptr.shape, new_idx.shape, new_ptr[-1])
    output = cxgc.tune_spmm(x, new_ptr, new_idx)


if __name__ == "__main__":
    # test_partition_by_degree()
    # test_partition_by_neighbor()
    test_single_submat()
