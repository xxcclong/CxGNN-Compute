import cxgnncomp as cxgc
import cxgnncomp_backend
import torch
import numpy


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
    # neighbor grouping
    neighbor_thres = 32

    def neighbor_grouping(ptr, neighbor_thres):
        ptr = ptr.cpu().numpy()
        new_ptr = [0]
        new_target = []
        for i in range(len(ptr) - 1):
            end = ptr[i + 1]
            start = ptr[i]
            while end - start > neighbor_thres:
                start += neighbor_thres
                new_ptr.append(start)
                new_target.append(i)
            new_ptr.append(end)
            new_target.append(i)
        return torch.from_numpy(numpy.array(new_ptr)).cuda(), torch.from_numpy(
            numpy.array(new_target)).cuda()

    new_ptr, new_target = neighbor_grouping(ptr, neighbor_thres)
    cxgc.prof(
        "spmm-multihead", "warm-up", lambda: cxgc.sage_sum_forward_edge_value(
            x, ptr, idx, val, ptr.shape[0] - 1))
    cxgc.prof(
        "spmm-multihead", "old", lambda: cxgc.sage_sum_forward_edge_value(
            x, ptr, idx, val, ptr.shape[0] - 1))
    cxgc.prof(
        "spmm-multihead",
        "old-neighbor-grouping", lambda: cxgc.sage_sum_forward_edge_value(
            x, new_ptr, idx, val, new_ptr.shape[0] - 1))
    cxgc.prof(
        "spmm-multihead", "pre-reduce",
        lambda: cxgnncomp_backend.spmm_multihead(
            ptr, idx, val, x, ptr.shape[0] - 1, cxgnncomp_backend.
            SPMM_MULTIHEAD_SCHEDULE.Optimal, 256))
    cxgc.prof(
        "spmm-multihead", "pre-reduce-neighbor-grouping",
        lambda: cxgnncomp_backend.spmm_multihead(
            new_ptr, idx, val, x, new_ptr.shape[0] - 1, cxgnncomp_backend.
            SPMM_MULTIHEAD_SCHEDULE.Optimal, 256))

    output = cxgc.tune_spmm(
        ptr.shape[0] - 1,
        idx.shape[0],
        x.shape[-1],
        cxgnncomp_backend.run_spmm_multihead_configurable,
        [ptr, idx, val, x, ptr.shape[0] - 1],
    )

    output = cxgc.tune_spmm(
        new_ptr.shape[0] - 1,
        idx.shape[0],
        x.shape[-1],
        cxgnncomp_backend.run_spmm_multihead_configurable,
        [new_ptr, idx, val, x, new_ptr.shape[0] - 1],
    )


if __name__ == "__main__":
    test_spmm_multihead()
    # tune_spmm_multihead()
    # test_spmm_multihead_neighbor_grouping()