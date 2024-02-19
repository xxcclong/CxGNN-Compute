import torch
import cxgnncomp as cxgc
from cxgnncomp import (
    MySageConv,
    MyGATConv,
    MyRGCNConv,
    MyGCNConv,
    LSTMConv,
    get_conv_from_str,
    get_model_from_str,
    Batch,
    PyGBatch,
    reorder_by
)
import dgl
import time


def train(model, params, label, optimizer, lossfn):
    torch.cuda.synchronize()
    t0 = time.time()
    optimizer.zero_grad()
    out = model(*params)
    if isinstance(model, LSTMConv):
        return
    loss = lossfn(out, label)
    torch.cuda.synchronize()
    t1 = time.time()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t2 = time.time()
    timer = cxgc.get_timers()
    print(f"forward {t1 - t0} backward {t2 - t1}")
    # timer.log_all(print)


def test_conv_training(args):
    infeat = args.infeat
    outfeat = args.outfeat
    num_head = args.num_head
    dev = torch.device("cuda:0")
    dset = args.dataset
    is_full_graph = args.is_full_graph

    feat, ptr, idx, b = cxgc.prepare_graph(
        dset=dset, feat_len=infeat, num_head=num_head, num_seeds=args.num_seeds
    )
    feat_label = torch.randn(
        [ptr.shape[0] - 1, outfeat], dtype=torch.float32, device=dev
    )
    feat.requires_grad_(True)

    cxgc.set_timers()

    conv = get_conv_from_str(args.model, infeat, outfeat, num_head).to(dev)
    conv.reset_parameters()
    optimizer = torch.optim.Adam(conv.parameters(), lr=0.1)
    lossfn = torch.nn.MSELoss()
    torch.cuda.synchronize()

    if isinstance(conv, MyGATConv):
        cxgc.prof(
            "train conv",
            args.model,
            lambda: train(
                conv,
                [
                    feat,
                    ptr,
                    idx,
                    b["num_node_in_layer"][-2],
                    b["num_node_in_layer"][-1],
                    idx.shape[0],
                ],
                feat_label,
                optimizer,
                lossfn,
            ),
        )
    elif isinstance(conv, MyRGCNConv):
        edge_types = torch.randint(0, conv.num_rel, (idx.shape[0],), device=dev)
        cxgc.prof(
            "train conv",
            args.model,
            lambda: train(
                conv,
                [feat, ptr, idx, edge_types, b["num_node_in_layer"][-2]],
                feat_label,
                optimizer,
                lossfn,
            ),
        )
    else:
        cxgc.prof(
            "train conv",
            args.model,
            lambda: train(
                conv,
                [feat, ptr, idx, b["num_node_in_layer"][-2]],
                feat_label,
                optimizer,
                lossfn,
            ),
        )
    torch.cuda.synchronize()


def get_dset_config(dset):
    if "arxiv" in dset:
        infeat = 128
        outfeat = 64
    elif dset == "products":
        infeat = 100
        outfeat = 64
    elif dset == "reddit":
        infeat = 602
        outfeat = 41
    elif "paper" in dset:
        infeat = 128
        outfeat = 256
    elif "friendster" in dset:
        infeat = 384
        outfeat = 64
    else:
        assert False, "unknown dataset"
    return infeat, outfeat


def get_model(args):
    mtype = args.model
    graph_type = args.graph_type
    hiddenfeat = args.hidden_feat
    num_layer = args.num_layer
    infeat, outfeat = get_dset_config(args.dataset)
    if mtype.upper() == "LSTM":
        infeat, outfeat = args.dedicate_feat, args.dedicate_feat
    dev = "cuda"
    num_head = args.num_head
    model = get_model_from_str(
        mtype,
        infeat,
        hiddenfeat,
        outfeat,
        graph_type,
        num_layer,
        num_head,
        args.num_rel,
        args.dataset,
        dropout=0,
    ).to(dev)
    return model


def to_dgl_block(ptr, idx, num_node_in_layer, num_edge_in_layer):
    print("num_node_in_layer", num_node_in_layer, num_edge_in_layer)
    blocks = []
    num_layer = num_node_in_layer.shape[0] - 1
    for i in range(len(num_node_in_layer) - 1):
        num_src = num_node_in_layer[num_layer - i]
        num_dst = num_node_in_layer[num_layer - i - 1]
        ptr = ptr[: num_dst + 1]
        idx = idx[: num_edge_in_layer[num_layer - i - 1]]
        blocks.append(
            dgl.create_block(
                ("csc", (ptr, idx, torch.tensor([]))), int(num_src), int(num_dst)
            )
        )
    return blocks


def run_model(args, model):
    dset = args.dataset
    infeat, outfeat = get_dset_config(args.dataset)
    if args.model.upper() == "LSTM":
        infeat, outfeat = args.dedicate_feat, args.dedicate_feat
    num_head = args.num_head
    dev = torch.device("cuda:0")
    feat, ptr, idx, b, edge_index = cxgc.prepare_graph(
        dset=dset,
        feat_len=infeat,
        num_head=num_head,
        num_seeds=args.num_seeds,
        is_full_graph=args.is_full_graph,
        need_edge_index=True,
    )
    if args.graph_type == "CSR_Layer":
        edge_index = None
        if args.model == "LSTM":
            deg = ptr[1:] - ptr[:-1]
            metric = torch.argsort(deg, descending=False)
            ptr, idx = reorder_by(ptr, idx, metric)
    feat_label = torch.randn(
        [b["num_node_in_layer"][0], outfeat], dtype=torch.float32, device=dev
    )
    # NG
    # new_ptr, new_target = cxgc.neighbor_grouping(ptr, neighbor_thres=32)
    # feat = torch.randn([new_ptr.shape[0] - 1, infeat],device=feat.device)
    # feat_label = torch.randn([new_ptr.shape[0] - 1, outfeat],device=feat.device)
    # ptr = new_ptr
    # b["num_node_in_layer"] = [ptr.shape[0] - 1 for i in range(len(b["num_node_in_layer"]))]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    lossfn = torch.nn.MSELoss()
    if args.graph_type == "CSR_Layer" or args.model == "LSTM":
        output = cxgc.prof(
            args.graph_type,
            args.model,
            lambda: train(
                model,
                [Batch(feat, ptr, idx, b["num_node_in_layer"], b["num_edge_in_layer"], edge_index=edge_index)],
                feat_label,
                optimizer,
                lossfn,
            ),
        )
    elif args.graph_type == "DGL":
        dgl_blocks = to_dgl_block(
            ptr, idx, b["num_node_in_layer"], b["num_edge_in_layer"]
        )
        output = cxgc.prof(
            args.graph_type,
            args.model,
            lambda: train(model, [[dgl_blocks, feat]], feat_label, optimizer, lossfn),
        )
    elif args.graph_type == "COO" or args.graph_type == "PyG":
        output = cxgc.prof(
            args.graph_type,
            args.model,
            lambda: train(
                model, [PyGBatch(feat, edge_index)], feat_label, optimizer, lossfn
            ),
        )
    else:
        assert False, "unknown graph type"
    print(f"ans {args.dataset} {args.model} {args.graph_type} {output[0]}")

    cxgc.global_tuner.save()


def test_model_training(args):
    cxgc.global_tuner.set_lazy(lazy=False)
    cxgc.set_timers()
    model = get_model(args)
    run_model(args, model)


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
    parser.add_argument("--dedicate_feat", type=int, default=32)
    args = parser.parse_args()
    print(args)
    if args.infeat > 0 and args.outfeat > 0:
        print("Benchmark single conv")
        test_conv_training(args)
    else:
        print("Benchmark model training")
        test_model_training(args)
