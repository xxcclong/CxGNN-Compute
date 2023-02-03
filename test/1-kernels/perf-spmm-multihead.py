import cxgnncomp as cxgc
import cxgnncomp_backend
import torch


def prepare_data():
    dset = "arxiv"
    infeat = 256
    num_head = 4
    x, ptr, idx, b = cxgc.prepare_data_full_graph(
        dset,
        feat_len=infeat,
        num_head=num_head,
    )
    # x, ptr, idx, b = cxgc.prepare_data_sampled_graph(dset=dset,
    #                                                  feat_len=infeat,
    #                                                  num_head=num_head,
    #                                                  num_seeds=1000)
    return x, ptr, idx, b, num_head


def test_spmm_multihead():
    # prepare data
    x, ptr, idx, b, num_head = prepare_data()
    val = torch.randn([idx.shape[0], num_head],
                      dtype=torch.float32,
                      device=x.device)
    print(x.shape, x.device, val.shape, val.device, ptr.shape, idx.shape)
    # cxgnncomp_backend.spmm_multihead(ptr, idx, val, x, ptr.shape[0] - 1,)
    # torch.cuda.synchronize()
    cxgc.prof(
        "spmm-multihead", "old", lambda: cxgc.sage_sum_forward_edge_value(
            x, ptr, idx, val, ptr.shape[0] - 1))
    cxgc.prof(
        "spmm-multihead", "pre-reduce",
        lambda: cxgnncomp_backend.spmm_multihead(
            ptr, idx, val, x, ptr.shape[0] - 1, cxgnncomp_backend.
            SPMM_MULTIHEAD_SCHEDULE.Optimal, 256))

    cxgc.prof(
        "spmm-multihead", "naive",
        lambda: torch.sum(cxgnncomp_backend.spmm_multihead(
            ptr, idx, val, x, ptr.shape[0] - 1, cxgnncomp_backend.
            SPMM_MULTIHEAD_SCHEDULE.Naive, 512),
                          dim=1))
    output1 = cxgnncomp_backend.spmm_multihead(
        ptr, idx, val, x, ptr.shape[0] - 1,
        cxgnncomp_backend.SPMM_MULTIHEAD_SCHEDULE.Naive, 512)
    output2 = cxgnncomp_backend.spmm_multihead(
        ptr, idx, val, x, ptr.shape[0] - 1,
        cxgnncomp_backend.SPMM_MULTIHEAD_SCHEDULE.Optimal, 256)
    cxgc.compare(torch.sum(output1, dim=1), output2)


def tune_spmm_multihead():
    # prepare data
    x, ptr, idx, b, num_head = prepare_data()
    val = torch.randn([idx.shape[0], num_head],
                      dtype=torch.float32,
                      device=x.device)

    outputs = cxgc.tune_spmm(
        ptr.shape[0] - 1,
        idx.shape[0],
        x.shape[-1],
        cxgnncomp_backend.run_spmm_multihead_configurable,
        [ptr, idx, val, x, ptr.shape[0] - 1],
    )
    output1 = cxgnncomp_backend.spmm_multihead(
        ptr, idx, val, x, ptr.shape[0] - 1,
        cxgnncomp_backend.SPMM_MULTIHEAD_SCHEDULE.Naive, 512)
    output2 = cxgc.tune_spmm(ptr.shape[0] - 1,
                             idx.shape[0],
                             x.shape[-1],
                             cxgnncomp_backend.run_spmm_multihead_configurable,
                             [ptr, idx, val, x, ptr.shape[0] - 1],
                             performance_param=outputs[1])[-1]
    cxgc.compare(torch.sum(output1, dim=1), output2)


def test_spmm_multihead_neighbor_grouping():
    # prepare data
    x, ptr, idx, b, num_head = prepare_data()
    val = torch.randn([idx.shape[0], num_head],
                      dtype=torch.float32,
                      device=x.device)
    x = torch.randn([x.shape[0], num_head, x.shape[-1]],
                    dtype=torch.float32,
                    device=x.device)


if __name__ == "__main__":
    test_spmm_multihead()
    tune_spmm_multihead()
