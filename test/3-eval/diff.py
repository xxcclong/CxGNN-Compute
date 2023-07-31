import torch
import cxgnncomp as cxgc
import cxgnncomp
import cxgnncomp_backend
import time
from torch_geometric.utils import softmax


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


def diff_gcn():

    thres = 64

    feat_len = 256
    num_head = 1
    torch.manual_seed(0)
    x, ptr, idx, batch, edge_index = cxgnncomp.prepare_data_full_graph(
        "arxiv", feat_len=feat_len, num_head=num_head, need_edge_index=True)

    ptr_ng, _ = cxgnncomp.neighbor_grouping(ptr, neighbor_thres=thres)
    num_node_in_layer = batch["num_node_in_layer"]
    tuner = cxgc.Tuner()

    original_time = cxgc.prof(
        "spmm", "origin", lambda: cxgnncomp_backend.sage_mean_forward(
            x, ptr, idx, ptr.shape[0] - 1))[0]
    output = tuner.tune_graph(ptr.shape[0] - 1, idx.shape[0], x.shape[1],
                              cxgnncomp_backend.run_spmm_configurable,
                              [ptr, idx, x, ptr.shape[0] - 1])

    cxgc.prof(
        "spmm", "origin-ng", lambda: cxgnncomp_backend.sage_mean_forward(
            x, ptr_ng, idx, ptr_ng.shape[0] - 1))
    output = tuner.tune_graph(ptr_ng.shape[0] - 1, idx.shape[0], x.shape[1],
                              cxgnncomp_backend.run_spmm_configurable,
                              [ptr_ng, idx, x, ptr_ng.shape[0] - 1])

    deg = ptr[1:] - ptr[:-1]
    print(torch.sum(deg <= thres), ptr.shape)
    ptr1, idx1 = cxgc.remove_from_graph(ptr, idx, deg > thres)

    cxgc.prof(
        "spmm", "<=thres", lambda: cxgnncomp_backend.sage_mean_forward(
            x, ptr1, idx1, ptr1.shape[0] - 1))
    output = tuner.tune_graph(ptr1.shape[0] - 1, idx1.shape[0], x.shape[1],
                              cxgnncomp_backend.run_spmm_configurable,
                              [ptr1, idx1, x, ptr1.shape[0] - 1])

    ptr1, idx1 = cxgc.remove_from_graph(ptr, idx, deg <= thres)
    cxgc.prof(
        "spmm", ">thres", lambda: cxgnncomp_backend.sage_mean_forward(
            x, ptr1, idx1, ptr1.shape[0] - 1))
    output = tuner.tune_graph(ptr1.shape[0] - 1, idx1.shape[0], x.shape[1],
                              cxgnncomp_backend.run_spmm_configurable,
                              [ptr1, idx1, x, ptr1.shape[0] - 1])

    ptr1, _ = cxgnncomp.neighbor_grouping(ptr1, neighbor_thres=thres)
    cxgc.prof(
        "spmm", ">thres, ng", lambda: cxgnncomp_backend.sage_mean_forward(
            x, ptr1, idx1, ptr1.shape[0] - 1))
    output = tuner.tune_graph(ptr1.shape[0] - 1, idx1.shape[0], x.shape[1],
                              cxgnncomp_backend.run_spmm_configurable,
                              [ptr1, idx1, x, ptr1.shape[0] - 1])

    weight = torch.randn([feat_len, feat_len], device="cuda")
    cxgc.prof("matmul", "matmul", lambda: torch.matmul(x, weight))


def diff_gat():

    feat_len = 256
    num_head = 1
    torch.manual_seed(0)
    x, ptr, idx, batch, edge_index = cxgnncomp.prepare_data_full_graph(
        "arxiv", feat_len=feat_len, num_head=num_head, need_edge_index=True)
    num_node_in_layer = batch["num_node_in_layer"]

    deg = ptr[1:] - ptr[:-1]
    print(torch.sum(deg <= 32), ptr.shape)
    ptr, idx = cxgc.remove_from_graph(ptr, idx, deg <= 32)
    num_node = ptr.shape[0] - 1

    att = torch.randn([idx.shape[0], num_head], device="cuda")
    att_src = torch.randn([num_node, num_head], device="cuda")
    att_dst = torch.randn([num_node, num_head], device="cuda")

    cxgnn_conv = cxgnncomp.MyGATConv(in_channels=feat_len,
                                     out_channels=feat_len,
                                     heads=num_head).cuda()
    cxgnncomp.prof("edge_softmax", "pyg", lambda: softmax(att, idx))
    cxgnncomp.prof(
        "edge_softmax", "cxgnn fused",
        lambda: cxgnn_conv.edge_softmax_fused(ptr=ptr,
                                              idx=idx,
                                              att_src=att_src,
                                              att_dst=att_dst,
                                              num_edge=idx.shape[0],
                                              relu_l=0.2))
    # cxgnncomp.prof(
    #     "edge_softmax", "cxgnn opwise",
    #     lambda: cxgnn_conv.edge_softmax_opwise(ptr=ptr,
    #                                            idx=idx,
    #                                            att_src=att_src,
    #                                            att_dst=att_dst,
    #                                            num_edge=idx.shape[0],
    #                                            relu_l=0.2))

    cxgnncomp.prof(
        "edge_softmax", "spmv", lambda: cxgnncomp_backend.spmv(
            ptr, idx, att_src.squeeze(), ptr.shape[0] - 1))

    for thres in [32, 64, 128, 256, 512]:
        new_ptr, new_target = cxgnncomp.neighbor_grouping(ptr,
                                                          neighbor_thres=thres)
        cxgnncomp.prof(
            "edge_softmax",
            "spmv-neighbor-group", lambda: cxgnncomp_backend.spmv(
                new_ptr, idx, att_src.squeeze(), new_ptr.shape[0] - 1))


def diff_lstm():
    torch.manual_seed(0)
    x, ptr, idx, b, num_head = prepare_data()
    deg = ptr[1:] - ptr[:-1]
    remove_flag = deg > 5000
    print(deg[remove_flag])
    ptr, idx = cxgc.remove_from_graph(ptr, idx, remove_flag)
    deg = ptr[1:] - ptr[:-1]
    print(torch.max(deg))
    count = torch.bincount(deg).cpu()
    in_feat = x.shape[-1]
    lstm_module = torch.nn.LSTM(in_feat, in_feat, batch_first=True).cuda()

    metric = torch.argsort(deg, descending=False)
    ptr, idx = cxgc.reorder_by(ptr, idx, metric)

    new_tensor = torch.randn([3, 13161, x.shape[-1]], device=x.device)

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    for num_center_in_batch in [256]:
        for _ in range(3):
            with torch.cuda.stream(s1):
                cxgc.NeighborLstmPadOP(lstm_module, ptr, idx, x, count,
                                       num_center_in_batch, 50000)
            with torch.cuda.stream(s2):
                lstm_module(new_tensor)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            with torch.cuda.stream(s2):
                lstm_module(new_tensor)
            with torch.cuda.stream(s1):
                cxgc.NeighborLstmPadOP(lstm_module, ptr, idx, x, count,
                                       num_center_in_batch, 50000)
            torch.cuda.synchronize()
        # torch.cuda.synchronize()
        print(f"batch size {num_center_in_batch} time {(time.time() - t0)/5}")
        cxgc.prof(
            "lstm neighbor op",
            f"padding {num_center_in_batch}", lambda: cxgc.NeighborLstmPadOP(
                lstm_module, ptr, idx, x, count, num_center_in_batch, 50000))
        cxgc.prof("lstm neighbor op2", f"padding {num_center_in_batch}",
                  lambda: lstm_module(new_tensor))

        for num_center_in_batch in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            cxgc.prof(
                "lstm neighbor op", f"padding {num_center_in_batch}",
                lambda: cxgc.NeighborLstmPadOP(lstm_module, ptr, idx, x, count,
                                               num_center_in_batch, 50000))


def diff_rgcn():
    remove = False
    torch.manual_seed(0)
    x, ptr, idx, b, num_head = prepare_data()
    num_node_origin = ptr.shape[0] - 1
    num_rel = 7
    if remove:
        remove_flag = torch.randn([idx.shape[0]], device=idx.device) > 0
        num_removed = int(torch.unique((idx[remove_flag])).shape[0])
        print(torch.sum(remove_flag), idx.shape)
        ptr, idx = cxgc.remove_from_graph_by_edge(ptr, idx, remove_flag)
        num_rel -= 1
        print(ptr.shape[0] - 1, num_removed, num_node_origin)
    num_center = ptr.shape[0] - 1
    num_edge = idx.shape[0]
    rel = torch.randint(0,
                        num_rel, [idx.shape[0]],
                        dtype=torch.int32,
                        device=x.device)
    rel_int64 = rel.to(torch.int64)
    weights = torch.randn([num_rel, x.shape[-1], x.shape[-1]],
                          dtype=torch.float32,
                          device=x.device)

    # s2e
    thres = 32
    preprocessed = cxgc.TypedLinearS2EOP.preprocess(weights,
                                                    rel_int64,
                                                    idx,
                                                    thres=thres)
    output1 = cxgc.TypedLinearS2EOP.apply(x, weights, rel_int64, idx,
                                          preprocessed)
    cxgc.prof("rgcn", "matmul", lambda: torch.mm(x, weights[0]))
    cxgc.prof("rgcn", "select matmul",
              lambda: cxgc.SelectMMS2EOP(x, weights, idx, rel))

    if remove:
        removed_tensor = torch.randn([num_removed, x.shape[-1]],
                                     device=x.device)
        cxgc.prof("rgcn", "matmul",
                  lambda: torch.mm(removed_tensor, weights[0]))


diff_lstm()
# diff_gat()
# diff_rgcn()
# diff_gcn()