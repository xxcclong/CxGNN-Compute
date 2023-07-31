import torch
import cxgnncomp as cxgc
import cxgnncomp_backend
import time
from torch.profiler import profile, record_function, ProfilerActivity


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


def test_spmm_matmul():
    torch.manual_seed(0)
    x, ptr, idx, b, num_head = prepare_data()
    # x = torch.ones_like(x)
    num_rel = 7
    num_center = ptr.shape[0] - 1
    num_edge = idx.shape[0]
    rel = torch.randint(0,
                        num_rel, [idx.shape[0]],
                        dtype=torch.int32,
                        device=x.device)
    rel_int64 = rel.to(torch.int64)
    single_weight = torch.randn([x.shape[-1], x.shape[-1]],
                                dtype=torch.float32,
                                device=x.device)
    dst = torch.repeat_interleave(
        torch.arange(ptr.shape[0] - 1, device=x.device), ptr[1:] - ptr[:-1])
    x_edge = x[idx]
    count = torch.bincount(rel, ).cpu()
    print(count)

    # weights = torch.repeat_interleave(single_weight.unsqueeze(0),
    #                                   num_rel,
    #                                   dim=0).reshape(num_rel, x.shape[-1],
    #                                                  x.shape[-1])
    weights = torch.randn([num_rel, x.shape[-1], x.shape[-1]],
                          dtype=torch.float32,
                          device=x.device)
    # weights = torch.ones_like(weights)

    # method 2
    output_dst1 = cxgc.TypedLinearS2DMMAggrOP.apply(x, weights, ptr, idx, rel,
                                                    ptr.shape[0] - 1)
    output_dst2 = cxgc.TypedLinearS2DAggrMMOP.apply(x, weights, ptr, idx, rel,
                                                    ptr.shape[0] - 1)
    output_dst3 = cxgc.TypedLinearS2DSort(x, weights, ptr, idx, rel,
                                          ptr.shape[0] - 1)
    output_dst4 = cxgc.TypedLinearS2DPushOP(x, weights, rel, idx, dst, num_rel,
                                            num_center, num_edge)
    print(
        "correct rate output_dst1 vs output_dst2:",
        torch.sum(torch.isclose(output_dst1, output_dst2, atol=1e-2,
                                rtol=1e-2)) / torch.numel(output_dst1))
    print(
        "correct rate output_dst1 vs output_dst3:",
        torch.sum(torch.isclose(output_dst1, output_dst3, atol=1e-2,
                                rtol=1e-2)) / torch.numel(output_dst1))
    print(
        "correct rate output_dst1 vs output_dst4:",
        torch.sum(torch.isclose(output_dst1, output_dst4, atol=1e-2,
                                rtol=1e-2)) / torch.numel(output_dst1))
    # print(output_dst1, output_dst4)
    output_edge1 = cxgc.TypedLinearE2EOP.apply(x_edge, weights, rel_int64)
    output_edge2 = cxgc.TypedLinearE2EOP.apply(x_edge, weights, rel_int64,
                                               False, count)
    output_edge3 = cxgc.TypedLinearS2EOP.apply(x, weights, rel_int64, idx)
    output_edge4 = cxgc.TypedLinearS2EOP.apply(x, weights, rel_int64, idx,
                                               False, count)
    print(
        "correct rate output_edge1 vs output_edge2:",
        torch.sum(
            torch.isclose(output_edge1, output_edge2, atol=1e-2, rtol=1e-2)) /
        torch.numel(output_edge1))
    print(
        "correct rate output_edge1 vs output_edge3:",
        torch.sum(
            torch.isclose(output_edge1 / (output_edge1[0] / output_edge3[0]),
                          output_edge3,
                          atol=1e-2,
                          rtol=1e-2)) / torch.numel(output_edge1))
    # print(output_edge1, output_edge3)
    print(
        "correct rate output_edge1 vs output_edge4:",
        torch.sum(
            torch.isclose(output_edge1, output_edge4, atol=1e-2, rtol=1e-2)) /
        torch.numel(output_edge1))

    cxgc.prof(
        "typed linear",
        "s2d aggr mm", lambda: cxgc.TypedLinearS2DMMAggrOP.apply(
            x, weights, ptr, idx, rel, ptr.shape[0] - 1))
    cxgc.prof(
        "typed linear",
        "s2d mm aggr", lambda: cxgc.TypedLinearS2DAggrMMOP.apply(
            x, weights, ptr, idx, rel, ptr.shape[0] - 1))
    cxgc.prof(
        "typed linear", "s2d sort", lambda: cxgc.TypedLinearS2DSort(
            x, weights, ptr, idx, rel, ptr.shape[0] - 1))

    cxgc.prof(
        "typed linear", "s2d push", lambda: cxgc.TypedLinearS2DPushOP(
            x, weights, rel, idx, dst, num_rel, num_center, num_edge))

    cxgc.prof("typed linear", "e2e",
              lambda: cxgc.TypedLinearE2EOP.apply(x_edge, weights, rel_int64))
    cxgc.prof(
        "typed linear", "e2e with count", lambda: cxgc.TypedLinearE2EOP.apply(
            x_edge, weights, rel_int64, False, count))
    cxgc.prof("typed linear", "s2e",
              lambda: cxgc.TypedLinearS2EOP.apply(x, weights, rel_int64, idx))
    cxgc.prof(
        "typed linear", "s2e with count", lambda: cxgc.TypedLinearS2EOP.apply(
            x, weights, rel_int64, idx, False, count))

    new_idx = torch.arange(0, idx.shape[0], device=idx.device)
    cxgc.prof(
        "aggregation", "e2d", lambda: cxgnncomp_backend.sage_sum_forward(
            output_edge1, ptr, new_idx, num_center))

    cxgc.prof(
        "aggregation", "s2d",
        lambda: cxgnncomp_backend.sage_sum_forward(x, ptr, idx, num_center))

    output_dst4 = torch.zeros([num_center, x.shape[-1]],
                              dtype=x.dtype,
                              device=x.device)
    output_dst4.index_add_(0, dst, output_edge1)
    print(
        "correct rate output_dst1 vs output_dst4:",
        torch.sum(torch.isclose(output_dst1, output_dst4, atol=1e-2,
                                rtol=1e-2)) / torch.numel(output_dst1))
    cxgc.prof("aggregation", "e2d index_add_",
              lambda: output_dst4.index_add_(0, dst, output_edge1))

    tuner = cxgc.Tuner()

    output = tuner.tune_graph(ptr.shape[0] - 1, idx.shape[0], x.shape[1],
                              cxgnncomp_backend.run_spmm_configurable,
                              [ptr, idx, x, ptr.shape[0] - 1])

    output = tuner.tune_graph(ptr.shape[0] - 1, new_idx.shape[0], x.shape[1],
                              cxgnncomp_backend.run_spmm_configurable,
                              [ptr, new_idx, output_edge1, ptr.shape[0] - 1])

    new_ptr, new_target = cxgc.neighbor_grouping(ptr, 32)

    output = tuner.tune_graph(new_ptr.shape[0] - 1, idx.shape[0], x.shape[1],
                              cxgnncomp_backend.run_spmm_configurable,
                              [new_ptr, idx, x, new_ptr.shape[0] - 1])

    output = tuner.tune_graph(
        new_ptr.shape[0] - 1, new_idx.shape[0], x.shape[1],
        cxgnncomp_backend.run_spmm_configurable,
        [new_ptr, new_idx, output_edge1, new_ptr.shape[0] - 1])

    output = tuner.tune_graph(new_ptr.shape[0] - 1, idx.shape[0], x.shape[1],
                              cxgnncomp_backend.run_spmm_configurable,
                              [new_ptr, idx, x, new_ptr.shape[0] - 1])


if __name__ == "__main__":
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        test_spmm_matmul()
    prof.export_chrome_trace("typed_linears_trace.json")