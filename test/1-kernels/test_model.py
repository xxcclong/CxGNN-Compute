import torch
import cxgnncomp as cxgc
from cxgnncomp import MySageConv, MyGATConv, MyRGCNConv
import cxgnncomp_backend
from torch.profiler import profile, record_function, ProfilerActivity


def train(model, params, label, optimizer, lossfn):
    optimizer.zero_grad()
    out = model(*params)
    loss = lossfn(out, label)
    loss.backward()
    optimizer.step()
    timer = cxgc.get_timers()
    timer.log_all(print)


def test_conv_training():
    infeat = 256
    outfeat = 256
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

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
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
                "train conv", "sage", lambda: train(conv, [
                    feat, ptr, idx, b["num_node_in_layer"][-2]
                ], feat_label, optimizer, lossfn))
    torch.cuda.synchronize()
    # output = cxgc.tune_spmm(ptr.shape[0] - 1, idx.shape[0], feat.shape[-1],
    #                         cxgnncomp_backend.run_spmm_configurable,
    #                         [ptr, idx, feat, ptr.shape[0] - 1])
    prof.export_chrome_trace("trace.json")


class Batch():

    def __init__(self, x, ptr, idx, num_node_in_layer, num_edge_in_layer=None):
        self.x = x
        self.ptr = ptr
        self.idx = idx
        self.num_node_in_layer = num_node_in_layer
        self.num_edge_in_layer = num_edge_in_layer


def test_model_training():
    infeat = 256
    hiddenfeat = 256
    outfeat = 128
    num_layer = 3
    num_head = 4
    dev = torch.device("cuda:0")

    dset = "arxiv"
    feat, ptr, idx, b = cxgc.prepare_data_full_graph(dset,
                                                     feat_len=infeat,
                                                     num_head=1)
    # feat, ptr, idx, b = cxgc.prepare_data_sampled_graph(
    #     dset=dset,
    #     feat_len=infeat,
    #     num_head=1,  # for model training, there are matmul expanding the heads
    #     num_seeds=1000)

    # model = cxgc.GAT(infeat,
    #                  hiddenfeat,
    #                  outfeat,
    #                  num_layer,
    #                  dropout=0.5,
    #                  graph_type="CSR_Layer",
    #                  config=None,
    #                  heads=num_head).to(dev)
    # model_name = "GAT"

    # model = cxgc.SAGE(infeat,
    #                   hiddenfeat,
    #                   outfeat,
    #                   num_layer,
    #                   dropout=0.5,
    #                   graph_type="CSR_Layer",
    #                   config=None).to(dev)
    # model_name = "SAGE"

    model = cxgc.GCN(infeat,
                     hiddenfeat,
                     outfeat,
                     num_layer,
                     dropout=0.5,
                     graph_type="CSR_Layer",
                     config=None).to(dev)
    model_name = "GCN"

    feat_label = torch.randn([b["num_node_in_layer"][0], outfeat],
                             dtype=torch.float32,
                             device=dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    lossfn = torch.nn.MSELoss()
    cxgc.prof(
        "train model", model_name, lambda: train(model, [
            Batch(feat, ptr, idx, b["num_node_in_layer"], b["num_edge_in_layer"
                                                            ])
        ], feat_label, optimizer, lossfn))


if __name__ == "__main__":
    test_conv_training()
    # test_model_training()