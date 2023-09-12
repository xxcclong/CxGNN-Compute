import torch
import cxgnncomp as cxgc
from cxgnncomp import MySageConv, MyGATConv, MyRGCNConv
import cxgnncomp_backend
from torch.profiler import profile, record_function, ProfilerActivity
import dgl
import time


def train(model, params, label, optimizer, lossfn):
    torch.cuda.synchronize()
    t0 = time.time()
    optimizer.zero_grad()
    out = model(*params)
    loss = lossfn(out, label)
    torch.cuda.synchronize()
    t1 = time.time()
    if not isinstance(model, cxgc.RGCN):
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    t2 = time.time()
    timer = cxgc.get_timers()
    print(f"forward {t1 - t0} backward {t2 - t1}")
    # timer.log_all(print)


def test_conv_training():
    infeat = 256
    outfeat = 32
    num_head = 4
    dev = torch.device("cuda:0")

    # dset = "arxiv"
    # feat, ptr, idx, b = cxgc.prepare_data_full_graph(dset,
    #                                                  feat_len=infeat,
    #                                                  num_head=1)
    dset = "papers100M"
    feat, ptr, idx, b = cxgc.prepare_data_sampled_graph(dset=dset,
                                                        feat_len=infeat,
                                                        num_head=1,
                                                        num_seeds=1000)
    feat_label = torch.randn([ptr.shape[0] - 1, outfeat],
                             dtype=torch.float32,
                             device=dev)
    feat.requires_grad_(True)

    cxgc.set_timers()

    # conv = MyGATConv(infeat, outfeat, heads=num_head).to(dev)
    # conv = MySageConv(infeat, outfeat).to(dev)
    # conv = cxgc.MyGCNConv(infeat, outfeat).to(dev)
    conv = MyRGCNConv(infeat, outfeat, num_rel=7).to(dev)
    conv.reset_parameters()
    optimizer = torch.optim.Adam(conv.parameters(), lr=0.1)
    lossfn = torch.nn.MSELoss()
    torch.cuda.synchronize()

    if isinstance(conv, cxgc.MyGATConv):
        cxgc.prof(
            "train conv", "gat", lambda: train(conv, [
                feat, ptr, idx, b["num_node_in_layer"][-2], b[
                    "num_node_in_layer"][-1], idx.shape[0]
            ], feat_label, optimizer, lossfn))
    elif isinstance(conv, MyRGCNConv):
        edge_types = torch.randint(0,
                                   conv.num_rel, (idx.shape[0], ),
                                   device=dev)
        cxgc.prof(
            "train conv", "rgcn", lambda: train(conv, [
                feat, ptr, idx, edge_types, b["num_node_in_layer"][-2]
            ], feat_label, optimizer, lossfn))
    else:
        cxgc.prof(
            "train conv", "sage",
            lambda: train(conv, [feat, ptr, idx, b["num_node_in_layer"][-2]],
                          feat_label, optimizer, lossfn))
    torch.cuda.synchronize()
    # output = cxgc.tune_spmm(ptr.shape[0] - 1, idx.shape[0], feat.shape[-1],
    #                         cxgnncomp_backend.run_spmm_configurable,
    #                         [ptr, idx, feat, ptr.shape[0] - 1])


class Batch():

    def __init__(self, x, ptr, idx, num_node_in_layer, num_edge_in_layer=None):
        self.x = x
        self.ptr = ptr
        self.idx = idx
        self.num_node_in_layer = num_node_in_layer
        self.num_edge_in_layer = num_edge_in_layer


class PyGBatch():

    def __init__(self, x, edge_index):
        self.x = x
        self.edge_index = edge_index


def get_dset_config(dset):
    if "arxiv" in dset:
        infeat = 128
        outfeat = 64
    elif dset == "products":
        infeat = 128
        outfeat = 64
    elif dset == "reddit":
        infeat = 128
        outfeat = 64
    elif "paper" in dset:
        infeat = 128
        outfeat = 64
    elif "friendster" in dset:
        infeat = 128
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
    dev = "cuda"
    num_head = args.num_head
    if mtype == "GCN":
        model = cxgc.GCN(infeat,
                         hiddenfeat,
                         outfeat,
                         num_layer,
                         dropout=0.5,
                         graph_type=graph_type,
                         config=None).to(dev)
    elif mtype == "GAT":
        model = cxgc.GAT(infeat,
                         hiddenfeat,
                         outfeat,
                         num_layer,
                         dropout=0.5,
                         graph_type=graph_type,
                         config=None,
                         heads=num_head).to(dev)
    elif mtype == "SAGE":
        model = cxgc.SAGE(infeat,
                          hiddenfeat,
                          outfeat,
                          num_layer,
                          dropout=0.5,
                          graph_type=graph_type,
                          config=None).to(dev)
    elif mtype == "RGCN":
        model = cxgc.RGCN(infeat,
                          hiddenfeat,
                          outfeat,
                          num_layer,
                          dropout=0.5,
                          graph_type=graph_type,
                          config=None,
                          num_rel=args.num_rel,
                          dataset_name=args.dataset).to(dev)
    else:
        assert False, "unknown model"
    return model


def to_dgl_block(ptr, idx, num_node_in_layer, num_edge_in_layer):
    blocks = []
    num_layer = num_node_in_layer.shape[0] - 1
    for i in range(len(num_node_in_layer) - 1):
        num_src = num_node_in_layer[num_layer - i]
        num_dst = num_node_in_layer[num_layer - i - 1]
        ptr = ptr[:num_dst + 1]
        idx = idx[:num_edge_in_layer[num_layer - i - 1]]
        blocks.append(
            dgl.create_block(('csc', (ptr, idx, torch.tensor([]))),
                             int(num_src), int(num_dst)))
    return blocks


def run_model(args, model):
    dset = args.dataset
    infeat, outfeat = get_dset_config(args.dataset)
    num_head = args.num_head
    dev = torch.device("cuda:0")
    feat, ptr, idx, b, edge_index = cxgc.prepare_data_full_graph(
        dset, feat_len=infeat, num_head=1, need_edge_index=1)
    feat_label = torch.randn([b["num_node_in_layer"][0], outfeat],
                             dtype=torch.float32,
                             device=dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    lossfn = torch.nn.MSELoss()
    if args.graph_type == "CSR_Layer":
        output = cxgc.prof(
            args.graph_type, args.model, lambda: train(model, [
                Batch(feat, ptr, idx, b["num_node_in_layer"], b[
                    "num_edge_in_layer"])
            ], feat_label, optimizer, lossfn))
    elif args.graph_type == "DGL":
        dgl_blocks = to_dgl_block(ptr, idx, b["num_node_in_layer"],
                                  b["num_edge_in_layer"])
        output = cxgc.prof(
            args.graph_type, args.model, lambda: train(
                model, [[dgl_blocks, feat]], feat_label, optimizer, lossfn))
    elif args.graph_type == "COO" or args.graph_type == "PyG":
        output = cxgc.prof(
            args.graph_type, args.model,
            lambda: train(model, [PyGBatch(feat, edge_index)], feat_label,
                          optimizer, lossfn))
    else:
        assert False, "unknown graph type"
    print(f"ans {args.dataset} {args.model} {args.graph_type} {output[0]}")
    # if args.model ==


def test_model_training():
    cxgc.set_timers()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--graph_type", type=str, default="CSR_Layer")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--hidden_feat", type=int, default=256)
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--num_head", type=int, default=1)
    parser.add_argument("--num_rel", type=int, default=7)
    args = parser.parse_args()
    print(args)
    model = get_model(args)
    run_model(args, model)


if __name__ == "__main__":
    # with profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # test_conv_training()
    test_model_training()
    # prof.export_chrome_trace("trace.json")
