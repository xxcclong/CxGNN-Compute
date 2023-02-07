import torch
import cxgnncomp as cxgc
import cxgnncomp_backend


def prepare_data():
    dset = "arxiv"
    infeat = 64
    num_head = 1
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


def test_spmm_matmul():
    x, ptr, idx, b, num_head = prepare_data()
    num_rel = 7
    rel = torch.randint(0,
                        num_rel, [idx.shape[0]],
                        dtype=torch.int32,
                        device=x.device)
    weights = torch.randn([num_rel, x.shape[-1], x.shape[-1]],
                          dtype=torch.float32,
                          device=x.device)
    # # method 1
    # cxgc.prof(
    #     "rgcn", "direct", lambda: cxgnncomp_backend.aggr_rgcn_direct_func(
    #         x, ptr, idx, weights, rel, ptr.shape[0] - 1))
    # cxgc.prof(
    #     "rgcn", "direct", lambda: cxgnncomp_backend.aggr_rgcn_direct_func(
    #         x, ptr, idx, weights, rel, ptr.shape[0] - 1))

    # # method 2
    # cxgc.prof(
    #     "rgcn", "pre-transform", lambda: cxgc.RGCNOP2.apply(
    #         x, weights, ptr, idx, rel, ptr.shape[0] - 1))

    # method 3
    dst = torch.repeat_interleave(
        torch.arange(ptr.shape[0] - 1, device=x.device), ptr[1:] - ptr[:-1])
    count = torch.bincount(rel, ).cpu()
    print(count)
    cxgc.prof(
        "rgcn", "sort mm", lambda: cxgc.RGCNOP_sorted(
            x,
            weights,
            idx,
            dst,
            count,
            ptr.shape[0] - 1,
        ))

    # profile sort
    def sort_and_move(x, rel):
        sorted, indices = torch.sort(rel)
        x = x[idx[indices]]
        count = torch.bincount(rel).cpu()

    cxgc.prof("rgcn", "sort", lambda: sort_and_move(x, rel))

    # method 4
    output = torch.empty([x.shape[0], weights.shape[-1]], device=x.device)
    v_rel = torch.randint(0, num_rel, [x.shape[0]], device=x.device)
    cxgc.prof(
        "rgcn", "typed_linear", lambda: cxgnncomp_backend.typed_linear(
            x, weights, output, v_rel.int(), 32))

    indexed_x = torch.index_select(x, 0, idx)
    output = torch.empty([indexed_x.shape[0], weights.shape[-1]],
                         device=x.device)
    cxgc.prof(
        "rgcn", "typed_linear", lambda: cxgnncomp_backend.typed_linear(
            indexed_x, weights, output, rel.int(), 32))

    cxgc.prof(
        "rgcn", "typed_linear", lambda: cxgnncomp_backend.typed_linear(
            indexed_x, weights, output, rel.int(), 64))


if __name__ == "__main__":
    test_spmm_matmul()