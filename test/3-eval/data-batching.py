import torch
import cxgnncomp as cxgc
import cxgnncomp_backend
import time


def prepare_data():
    infeat = 256
    num_head = 1

    dset = "arxiv"
    x, ptr, idx, b = cxgc.prepare_data_full_graph(
        dset,
        feat_len=infeat,
        num_head=num_head,
    )

    # dset = "papers100M"
    # x, ptr, idx, b = cxgc.prepare_data_sampled_graph(dset=dset,
    #                                                  feat_len=infeat,
    #                                                  num_head=num_head,
    #                                                  num_seeds=1000)

    return x, ptr, idx, b, num_head


def batching_rgcn():
    torch.manual_seed(0)
    x, ptr, idx, b, num_head = prepare_data()
    num_rel = 7
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
    output = torch.empty([num_edge, weights.shape[-1]], device=x.device)
    cxgc.prof(
        "rgcn", "single edge", lambda: cxgnncomp_backend.typed_linear_s2e(
            x, weights, output, idx, rel, 32))
    for padding in [32, 64, 128, 256, 512]:
        cxgc.prof(
            "rgcn", "single edge", lambda: cxgnncomp_backend.typed_linear_s2e(
                x, weights, output, idx, rel, padding))

    # for thres in [32, 64, 128, 256, 512]:
    #     preprocessed = cxgc.TypedLinearS2EOP.preprocess(weights,
    #                                                     rel_int64,
    #                                                     idx,
    #                                                     thres=thres)
    #     output2 = cxgc.TypedLinearS2EOP.apply(x, weights, rel_int64, idx,
    #                                           preprocessed)
    #     cxgc.compare(output1, output2)
    #     cxgc.prof(
    #         "rgcn batching", f"{thres}", lambda: cxgc.TypedLinearS2EOP.apply(
    #             x, weights, rel_int64, idx, preprocessed))
    exit()

    # e2e
    print("==========E2E==========")
    on_edge_x = x[idx]
    thres = 32
    preprocessed = cxgc.TypedLinearE2EOP.preprocess(weights,
                                                    rel_int64,
                                                    thres=thres)
    output1 = cxgc.TypedLinearE2EOP.apply(on_edge_x, weights, rel_int64,
                                          preprocessed)
    for thres in [32, 64, 128, 256, 512]:
        preprocessed = cxgc.TypedLinearE2EOP.preprocess(weights,
                                                        rel_int64,
                                                        thres=thres)
        output2 = cxgc.TypedLinearE2EOP.apply(on_edge_x, weights, rel_int64,
                                              preprocessed)
        cxgc.compare(output1, output2)
        cxgc.prof(
            "rgcn batching", f"{thres}", lambda: cxgc.TypedLinearE2EOP.apply(
                on_edge_x, weights, rel_int64, preprocessed))


def batching_lstm():
    torch.manual_seed(0)
    x, ptr, idx, b, num_head = prepare_data()
    deg = ptr[1:] - ptr[:-1]
    count = torch.bincount(deg).cpu()
    in_feat = x.shape[-1]
    lstm_module = torch.nn.LSTM(in_feat, in_feat, batch_first=True).cuda()

    metric = torch.argsort(deg, descending=False)
    ptr, idx = cxgc.reorder_by(ptr, idx, metric)
    # cxgc.prof("lstm neighbor op", "padding",
    #           lambda: cxgc.NeighborLstmOP(lstm_module, ptr, idx, x, count))
    for num_center_in_batch in [
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
            16384, 32768, 65536
    ]:
        cxgc.prof(
            "lstm neighbor op",
            f"padding {num_center_in_batch}", lambda: cxgc.NeighborLstmPadOP(
                lstm_module, ptr, idx, x, count, num_center_in_batch, 50000))


# batching_rgcn()
batching_lstm()