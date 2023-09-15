import torch
import cxgnncomp as cxgc
import time
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, world_size, args):
    feat, ptr, idx, b, edge_index = cxgc.prepare_graph(
        dset=args.dataset,
        feat_len=args.infeat,
        num_head=args.num_head,
        num_seeds=args.num_seeds,
        need_edge_index=True,
        is_full_graph=args.is_full_graph,
        need_feat=False)
    num_node = ptr.shape[0] - 1

    model = cxgc.get_model_from_str(args.model,
                                    args.infeat,
                                    args.hiddenfeat,
                                    args.outfeat,
                                    args.graph_type,
                                    num_layer=args.num_layer,
                                    num_head=args.num_head,
                                    num_rel=args.num_rel,
                                    dataset=args.dataset).to(rank)

    # reorder graph for locality
    if args.reorder_file != "":
        new_order = torch.from_numpy(
            np.fromfile(args.reorder_file + "reorder_" + args.dataset +
                        ".dat")).to(rank)
        # reorder ptr and idx
        raise NotImplementedError

    # prepare local features
    if rank != world_size - 1:
        local_feat = torch.randn([num_node // world_size,
                                  args.infeat]).to(rank)
    else:
        local_feat = torch.randn(
            [num_node // world_size + (num_node % world_size),
             args.infeat]).to(rank)
    graphs = []
    for remote_rank in range(args.num_device):
        ptr, idx, target = cxgc.partition_2d_gpu(edge_index, args.num_device,
                                                 rank, remote_rank)
        graphs.append(cxgc.Batch(ptr=ptr, idx=idx, target=target))
    for item in graphs:
        item.to(rank)


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
    mp.spawn(run,
             args=(args.num_device, args),
             nprocs=args.num_device,
             join=True)
