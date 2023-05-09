import cxgnncomp as cxgc
import cxgnncomp
import torch
import time
import argparse
import numpy as np
import os
import sys
import cxgnncomp_backend


def prepare_data():
    infeat = 256
    num_head = 1
    dset = "arxiv"
    x, ptr, idx, b, edge_index = cxgc.prepare_data_full_graph(
        dset, feat_len=infeat, num_head=num_head, need_edge_index=True)
    return x, ptr, idx, b, num_head, edge_index


x, ptr, idx, b, num_head, edge_index = prepare_data()
deg = ptr[1:] - ptr[:-1]


def show_tune_stages_gcn():
    num_edge = idx.shape[0]
    ans = []
    # stage 1: vertex centric
    t = cxgc.prof(
        "spmm", "origin", lambda: cxgnncomp_backend.sage_mean_forward(
            x, ptr, idx, ptr.shape[0] - 1))[0]
    ans.append(t)
    # stage 2: neighbor grouping
    for thres in [512, 256, 128, 64, 32]:
        new_ptr, new_target = cxgnncomp.neighbor_grouping(ptr,
                                                          neighbor_thres=thres)
        t = cxgc.prof(
            "spmm", f"ng-{thres}", lambda: cxgnncomp_backend.sage_mean_forward(
                x, new_ptr, idx, new_ptr.shape[0] - 1))[0]
        ans.append(t)
    # stage 3: differentiaed exec

    thres = 32
    ptr1, idx1 = cxgc.remove_from_graph(ptr, idx, deg <= thres)
    new_ptr1, new_target1 = cxgnncomp.neighbor_grouping(ptr1,
                                                        neighbor_thres=thres)
    ptr2, idx2 = cxgc.remove_from_graph(ptr, idx, deg > thres)

    t0 = cxgc.prof(
        "spmm", f"ng <={thres}", lambda: cxgnncomp_backend.sage_mean_forward(
            x, ptr2, idx2, ptr2.shape[0] - 1))[0]
    t1 = cxgc.prof(
        "spmm", f"ng >{thres}", lambda: cxgnncomp_backend.sage_mean_forward(
            x, new_ptr1, idx1, new_ptr1.shape[0] - 1))[0]
    print("differentiated", t0 + t1)
    ans.append(t0 + t1)

    tuner = cxgc.Tuner()

    outputs = tuner.tune_graph(new_ptr1.shape[0] - 1,
                               idx1.shape[0],
                               x.shape[1],
                               cxgnncomp_backend.run_spmm_configurable,
                               [new_ptr1, idx1, x, new_ptr1.shape[0] - 1],
                               verbose=True,
                               rettime=True)
    for item in outputs:
        ans.append(t0 + item)

    # t1_new = cxgc.prof(
    #     "spmm", f"ng >{thres} tuned", lambda: tuner.tune_graph(
    #         new_ptr1.shape[0] - 1, idx1.shape[0], x.shape[1], cxgnncomp_backend
    #         .run_spmm_configurable, [new_ptr1, idx1, x, new_ptr1.shape[0] - 1])
    # )[0]
    # print("differentiated", t0 + t1_new)

    print(ans)
    for item in ans:
        print(num_edge / item)


def show_tune_stages_rgcn():
    ans = []
    num_rel = 7
    num_type = num_rel
    infeat = x.shape[-1]
    dev = x.device

    weight = torch.randn([num_type, infeat, infeat], device=dev)
    types = torch.randint(0,
                          num_type, [edge_index.shape[1]],
                          device=x.device,
                          dtype=torch.int32)
    rel = types
    rel_int64 = rel.to(torch.int64)

    # stage 1: vertex centric
    t = cxgc.prof(
        "mlp", "vertex", lambda: cxgnncomp_backend.aggr_rel_direct(
            x, ptr, idx, weight, rel, ptr.shape[0] - 1, num_rel))[0]
    ans.append(t)
    # stage 3: neighbor grouping
    for thres in [512, 256, 128, 64, 32]:
        new_ptr, new_target = cxgnncomp.neighbor_grouping(ptr,
                                                          neighbor_thres=thres)
        t = cxgc.prof(
            "mlp", f"ng{thres}", lambda: cxgnncomp_backend.aggr_rel_direct(
                x, new_ptr, idx, weight, rel, new_ptr.shape[0] - 1, num_rel)
        )[0]
        ans.append(t)
    # stage 4: differentiaed exec
    remove_flag = torch.randn([idx.shape[0]], device=idx.device) > 0
    num_removed = int(torch.unique((idx[remove_flag])).shape[0])
    print(torch.sum(remove_flag), idx.shape)
    part_ptr, part_idx = cxgc.remove_from_graph_by_edge(ptr, idx, remove_flag)
    part_rel_int64 = rel_int64[~remove_flag]
    assert part_rel_int64.shape == part_idx.shape, (part_rel_int64.shape,
                                                    part_idx.shape)
    removed_tensor = torch.randn([num_removed, x.shape[-1]], device=x.device)
    tmm = cxgc.prof("rgcn", "matmul",
                    lambda: torch.mm(removed_tensor, weight[0]))[0]

    for thres in [512, 256, 128, 64, 32]:
        new_ptr, new_target = cxgnncomp.neighbor_grouping(part_ptr,
                                                          neighbor_thres=thres)
        t = cxgc.prof(
            "mlp", f"ng{thres}part", lambda: cxgnncomp_backend.aggr_rel_direct(
                x, new_ptr, part_idx, weight, rel, new_ptr.shape[0] - 1,
                num_rel))[0]
        # ans.append(t + tmm)

    # stage 2: edge centric
    output = torch.zeros_like(x)
    t = cxgc.prof(
        "mlp", "edge", lambda: cxgnncomp_backend.typed_linear_s2d(
            x, weight, output, edge_index[0], edge_index[1], types, 32))[0]
    ans.append(t)

    # batched mm
    thres = 32
    preprocessed = cxgc.TypedLinearS2EOP.preprocess(weight,
                                                    rel_int64,
                                                    idx,
                                                    thres=thres)

    ans.append(
        cxgc.prof(
            "mlp", "batched", lambda: cxgc.TypedLinearS2EOP.apply(
                x, weight, rel_int64, idx, preprocessed))[0])

    print(part_idx.shape, part_rel_int64.shape)
    part_preprocessed = cxgc.TypedLinearS2EOP.preprocess(weight,
                                                         part_rel_int64,
                                                         part_idx,
                                                         thres=thres)
    torch.cuda.synchronize()
    ans.append(
        cxgc.prof(
            "mlp", "batched-part", lambda: cxgc.TypedLinearS2EOP.apply(
                x, weight, part_rel_int64, part_idx, part_preprocessed))[0] +
        tmm)

    print(ans)
    for item in ans:
        print(item)


show_tune_stages_gcn()
# show_tune_stages_rgcn()