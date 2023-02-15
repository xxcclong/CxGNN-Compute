import torch
import cxgnncomp_backend
import cxgnncomp

x, ptr, idx, batch = cxgnncomp.prepare_data_sampled_graph("rmag240m", 10000)
torch.cuda.synchronize()
rel = cxgnncomp_backend.gen_edge_type_mag240m(ptr, idx, batch["sub_to_full"])
torch.cuda.synchronize()
print(torch.max(rel), torch.min(rel))
print(rel.dtype)
torch.cuda.synchronize()