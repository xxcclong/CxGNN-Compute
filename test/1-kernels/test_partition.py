import torch
import cxgnncomp as cxgc
import time
import numpy as np


def run(args):
    feat, ptr, idx, b, edge_index = cxgc.prepare_graph(
        dset=args.dataset,
        feat_len=args.infeat,
        num_head=args.num_head,
        num_seeds=args.num_seeds,
        need_edge_index=1,
        is_full_graph=args.is_full_graph,
        need_feat=False,
        device="cpu")
    # rank_dst = 0
    num_edge = 0
    num_total_edge = idx.shape[0]
    for i in range(args.num_device):
        for j in range(args.num_device):
            ptr, idx, target = cxgc.partition_2d_gpu(edge_index, args.num_device, i, j)
            print(ptr.shape, idx.shape, target.shape)
            num_edge += idx.shape[0]
    assert num_edge == num_total_edge, f"{num_edge} {num_total_edge}"




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--graph_type", type=str, default="CSR_Layer")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--hidden_feat", type=int, default=256)
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--num_head", type=int, default=1)
    parser.add_argument("--num_rel", type=int, default=7)
    parser.add_argument("--infeat", type=int, default=-1)
    parser.add_argument("--outfeat", type=int, default=-1)
    parser.add_argument("--is_full_graph", type=int, default=1)
    parser.add_argument("--num_seeds", type=int, default=1000)
    parser.add_argument("--reorder_file",
                        type=str,
                        default="/home/huangkz/repos/rabbit_order/demo/")
    parser.add_argument("--num_device", type=int, default=4)
    args = parser.parse_args()
    print(args)
    run(args)