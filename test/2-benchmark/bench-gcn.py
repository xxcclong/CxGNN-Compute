'''
Testing graph-prioritized and nn-priroritized scheduling for GCN,
showing that for models with simple 
'''

import torch
import cxgnncomp as cxgc
import cxgnncomp_backend
import time


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
dst = torch.repeat_interleave(torch.arange(ptr.shape[0] - 1, device=x.device),
                              ptr[1:] - ptr[:-1])
num_edge = idx.shape[0]
num_center = ptr.shape[0] - 1
val = torch.randn([num_edge], dtype=torch.float32, device=x.device)


def bench_graph_gcn():
    output = cxgc.prof("gcn",
                       "graph",
                       lambda: cxgnncomp_backend.sage_sum_forward_edge_value(
                           x, ptr, idx, val, num_center),
                       display=False)
    return output


def run_neural_gcn():
    output = torch.zeros([num_center, x.shape[-1]],
                         dtype=torch.float32,
                         device=x.device)
    torch.cuda.synchronize()
    t0 = time.time()
    expanded_x = x[idx]  # s2e
    torch.cuda.synchronize()
    t1 = time.time()
    expanded_x = expanded_x * val.view(-1, 1)
    torch.cuda.synchronize()
    t2 = time.time()
    output.index_add_(0, dst, expanded_x)
    torch.cuda.synchronize()
    t3 = time.time()
    return output, t2 - t1, t3 - t2 + t1 - t0


def bench_neural_gcn():
    output = cxgc.prof("gcn",
                       "neural",
                       lambda: run_neural_gcn(),
                       display=False)
    output_tensor, tnn, tgraph = run_neural_gcn()
    return output, tnn, tgraph


feat = 32
feats = []
for i in range(10):
    feats.append(feat)
    feat *= 2
    if feat > 512:
        break
for feat in feats:
    x = torch.randn(x.shape[0], feat, device=x.device, dtype=x.dtype)
    output_graph = bench_graph_gcn()
    output_nn, tnn, tgraph = bench_neural_gcn()
    print(output_graph[0], tnn * 1000, tgraph * 1000, output_nn[0])
