import dgl
import torch
import cxgnncomp as cxgc
import dgl.function as fn
import time

a = torch.randn(1024, 1024, device='cuda')
b = torch.randn(1024, 1024, device='cuda')
for _ in range(3):
    c = torch.matmul(a, b)
torch.cuda.synchronize()
start = time.time()
for _ in range(1):
    c = torch.matmul(a, b)
torch.cuda.synchronize()
print(time.time() - start)

dset = "arxiv"
x, ptr, idx, batch, edge_index = cxgc.prepare_data_full_graph(
    dset=dset,
    need_edge_index=True,
    need_feat=True,
    undirected=False,
    feat_len=800)
g = dgl.DGLGraph((edge_index[0], edge_index[1])).to("cuda")

print(x.shape)

print(idx.shape[0] * x.shape[1])

g.ndata['ft'] = x
start = time.time()
for _ in range(1):
    g.update_all(fn.copy_src(src='ft', out='m'), fn.sum(msg='m', out='ft'))
torch.cuda.synchronize()
print(time.time() - start)