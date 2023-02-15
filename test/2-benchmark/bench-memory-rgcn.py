'''
testing different ways of RGCN, showing that graph-prioritized scheduing is good with small features
while neural-prioritized scheduling is good with large features
So, although the graph-prioritized scheduling is optimal for simple graph convolutions, 
it is not optimal for performing neural operations such as attention, MLP, transformer, difussion with graph structure
'''

import torch
import cxgnncomp as cxgc
import cxgnncomp_backend


def prepare_data():
    dset = "arxiv"
    infeat = 64
    num_head = 1
    # x, ptr, idx, b = cxgc.prepare_data_full_graph(
    #     dset,
    #     feat_len=infeat,
    #     num_head=num_head,
    # )
    x, ptr, idx, b = cxgc.prepare_data_sampled_graph(dset=dset,
                                                     feat_len=infeat,
                                                     num_head=num_head,
                                                     num_seeds=1000)
    return x, ptr, idx, b, num_head


x, ptr, idx, b, num_head = prepare_data()
x.requires_grad_(True)
dst = torch.repeat_interleave(torch.arange(ptr.shape[0] - 1, device=x.device),
                              ptr[1:] - ptr[:-1])
num_rel = 7
num_edge = idx.shape[0]
# print(x.shape, x.device, ptr.shape, idx.shape, num_rel, num_edge)
rel = torch.randint(0,
                    num_rel, [idx.shape[0]],
                    dtype=torch.int32,
                    device=x.device)
weights = torch.randn([num_rel, x.shape[-1], x.shape[-1]],
                      dtype=torch.float32,
                      device=x.device,
                      requires_grad=True)
num_edge = idx.shape[0]
num_center = ptr.shape[0] - 1
count = torch.bincount(rel).cpu()


def bench_naive_rgcn():
    base = torch.cuda.memory_allocated(0) / 1e6
    output = cxgc.TypedLinearNaiveS2D(x, weights, rel, idx, dst, num_center,
                                      num_edge)
    # print(torch.cuda.memory_allocated(0) / 1e6 - base)


def bench_nn_prior_rgcn():
    base = torch.cuda.memory_allocated(0) / 1e6
    output = cxgc.TypedLinearS2DSort(x,
                                     weights,
                                     ptr,
                                     idx,
                                     rel,
                                     ptr.shape[0] - 1,
                                     count=count)
    # print(torch.cuda.memory_allocated(0) / 1e6 - base)
    output = cxgc.prof(
        "rgcn", "nn prior", lambda: cxgc.TypedLinearS2DSort(
            x, weights, ptr, idx, rel, ptr.shape[0] - 1, count=count))
    output_tensor, tgraph, tnn, tother = cxgc.TypedLinearS2DSort(x,
                                                                 weights,
                                                                 ptr,
                                                                 idx,
                                                                 rel,
                                                                 ptr.shape[0] -
                                                                 1,
                                                                 count=count)
    return output, tgraph, tnn, tother


def bench_graph_prior_rgcn():
    base = torch.cuda.memory_allocated(0) / 1e6
    output = cxgc.RGCNOP3.apply(x, weights, ptr, idx, rel, ptr.shape[0] - 1)
    # print(torch.cuda.memory_allocated(0) / 1e6 - base)
    cxgc.prof(
        "rgcn", "graph prior", lambda: cxgc.RGCNOP3.apply(
            x, weights, ptr, idx, rel, ptr.shape[0] - 1))


def bench_graph_prior_rgcn2():
    base = torch.cuda.memory_allocated(0) / 1e6
    output = torch.empty([num_center, weights.shape[-1]], device=x.device)
    cxgnncomp_backend.typed_linear_s2d(x, weights, output, idx, dst, rel.int(),
                                       32)
    # print(torch.cuda.memory_allocated(0) / 1e6 - base)
    return cxgc.prof(
        "rgcn", "graph prior2", lambda: cxgnncomp_backend.typed_linear_s2d(
            x, weights, output, idx, dst, rel.int(), 32))


feat = 32
feats = []
for i in range(10):
    feats.append(feat)
    feat *= 2
    if feat > 512:
        break
for feat in feats:
    x = torch.randn(x.shape[0], feat, device=x.device, dtype=x.dtype)
    weights = torch.randn([num_rel, x.shape[-1], x.shape[-1]],
                          dtype=torch.float32,
                          device=x.device,
                          requires_grad=True)
    # bench_naive_rgcn()
    output_nn, tgraph, tnn, tother = bench_nn_prior_rgcn()
    # bench_graph_prior_rgcn()
    output_graph = bench_graph_prior_rgcn2()
    print(output_nn[0], tgraph, tnn, tother, output_graph[0])