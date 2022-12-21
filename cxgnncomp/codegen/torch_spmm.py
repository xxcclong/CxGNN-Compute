import torch


def spmm_torch(x: torch.Tensor, ptr: torch.Tensor, idx: torch.Tensor, num_nodes: int, val=None):
    if val is None:
        val = torch.ones_like(idx, dtype=torch.float32)
    sparse_mat = torch.sparse_csr_tensor(ptr, idx, val, device=ptr.device)
    output = torch.sparse.mm(sparse_mat, x)
    return output
