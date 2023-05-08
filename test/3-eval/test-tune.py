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
    x, ptr, idx, b = cxgc.prepare_data_full_graph(
        dset,
        feat_len=infeat,
        num_head=num_head,
    )
    return x, ptr, idx, b, num_head


x, ptr, idx, b, num_head = prepare_data()
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


show_tune_stages_gcn()