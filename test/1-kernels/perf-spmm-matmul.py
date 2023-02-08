import torch
import cxgnncomp as cxgc
import cxgnncomp_backend
import time


def prepare_data():
    dset = "arxiv"
    infeat = 256
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
    num_edge = idx.shape[0]
    rel = torch.randint(0,
                        num_rel, [idx.shape[0]],
                        dtype=torch.int32,
                        device=x.device)
    single_weight = torch.randn([x.shape[-1], x.shape[-1]],
                                dtype=torch.float32,
                                device=x.device)

    weights = torch.repeat_interleave(single_weight.unsqueeze(0),
                                      num_rel,
                                      dim=0).reshape(num_rel, x.shape[-1],
                                                     x.shape[-1])
    # weights = torch.randn([num_rel, x.shape[-1], x.shape[-1]],
    #                       dtype=torch.float32,
    #                       device=x.device)
    # # method 1
    # cxgc.prof(
    #     "rgcn", "direct", lambda: cxgnncomp_backend.aggr_rgcn_direct_func(
    #         x, ptr, idx, weights, rel, ptr.shape[0] - 1))
    # cxgc.prof(
    #     "rgcn", "direct", lambda: cxgnncomp_backend.aggr_rgcn_direct_func(
    #         x, ptr, idx, weights, rel, ptr.shape[0] - 1))

    # method 2
    gt_output_dst = cxgc.RGCNOP2.apply(x, weights, ptr, idx, rel,
                                       ptr.shape[0] - 1)
    cxgc.prof(
        "rgcn", "pre-transform", lambda: cxgc.RGCNOP2.apply(
            x, weights, ptr, idx, rel, ptr.shape[0] - 1))

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
        # count = torch.bincount(rel).cpu()

    def sort_only(x, rel):
        sorted, indices = torch.sort(rel)

    cxgc.prof("rgcn", "sort and move", lambda: sort_and_move(x, rel))

    cxgc.prof("rgcn", "sort only", lambda: sort_only(x, rel))

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

    gt_output_edge = torch.empty([indexed_x.shape[0], weights.shape[-1]],
                                 device=x.device)
    cxgnncomp_backend.typed_linear(indexed_x, weights, gt_output_edge,
                                   rel.int(), 64)

    # method 5
    count = torch.bincount(rel, ).cpu()

    def aligh_rel(rel, count, thres):
        arr = []
        for it, r in enumerate(count):
            if r % thres != 0:
                toadd = thres - r % thres
                arr += ([it for _ in range(toadd)])
        arr = torch.Tensor(arr).cuda()
        return torch.cat([rel, arr], dim=0)

    cxgc.prof("rgcn", "align-rel", lambda: aligh_rel(rel, count, 32))

    # method 6
    def new_typed_linear(x, weights, old_idx, old_rel, num_edge):
        thres = 256
        count = torch.bincount(old_rel, )
        rel = torch.empty([old_rel.shape[0] + num_rel * (thres - 1)],
                          device=x.device,
                          dtype=old_rel.dtype)
        rel[:old_rel.shape[0]] = old_rel
        idx = torch.empty([old_idx.shape[0] + num_rel * (thres - 1)],
                          device=x.device,
                          dtype=old_idx.dtype)
        idx[:old_idx.shape[0]] = old_idx
        # rel = old_rel.resize_(num_edge + num_rel * (thres - 1))
        # idx = old_idx.resize_(num_edge + num_rel * (thres - 1))
        cxgnncomp_backend.pad_rel(rel, idx, count, thres, num_rel, num_edge)
        sorted_rel, indices = torch.sort(rel)
        # cxgc.prof(
        #     "rgcn", "typed_matmul", lambda: cxgc.codegen.typed_matmul(
        #         x, weights, idx[indices], indices, sorted_rel))
        # print(idx[indices])
        output = cxgc.codegen.typed_matmul(x, weights, idx[indices], indices,
                                           sorted_rel)

        return output

    rel = rel.to(torch.int64)
    output_edge = new_typed_linear(x, weights, idx, rel, num_edge=idx.shape[0])
    print(
        torch.allclose(output_edge[:num_edge],
                       gt_output_edge,
                       atol=1e-3,
                       rtol=1e-2))
    print(output_edge[:num_edge][torch.isclose(
        output_edge[:num_edge], gt_output_edge, atol=1e-3, rtol=1e-2) ==
                                 False])
    print(gt_output_edge[torch.isclose(
        output_edge[:num_edge], gt_output_edge, atol=1e-3, rtol=1e-2) ==
                         False])
    print(gt_output_edge[torch.isclose(
        output_edge[:num_edge], gt_output_edge, atol=1e-3, rtol=1e-2) ==
                         False].shape)
    print(gt_output_edge.shape)
    # print(output_edge[:num_edge], gt_output_edge)
    # print(torch.matmul(x, weights[0]))
    # print(torch.matmul(x[idx], weights[0]))
    # print(torch.matmul(x[0], weights[0]))
    # print(idx)

    cxgc.prof(
        "rgcn", "new_typed_linear",
        lambda: new_typed_linear(x, weights, idx, rel, num_edge=idx.shape[0]))


if __name__ == "__main__":
    test_spmm_matmul()