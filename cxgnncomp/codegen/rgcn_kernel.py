import triton
import torch
import triton.language as tl
import cxgnncomp_backend

config_arr = [
    # triton.Config({
    #     'BLOCK_X': 32,
    #     'BLOCK_Y': 2,
    #     'BLOCK_Z': 128
    # }),
    triton.Config({
        'BLOCK_X': 32,
        'BLOCK_Y': 2,
        'BLOCK_Z': 64
    }),
]


@triton.autotune(configs=config_arr, key=["feat_len"])
@triton.jit
def rgcn_kernel(
    input,
    ptr,
    idx,
    rel,
    weights,
    output,
    feat_len,
    out_feat_len: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    BLOCK_Z: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # * BLOCK_X // 32
    pid_y = tl.program_id(axis=1)
    pid_z = tl.program_id(axis=2)
    neighbor_start = tl.load(ptr + pid)
    neighbor_end = tl.load(ptr + pid + 1)
    offset_output = tl.arange(0, 1)
    for ydim in range(BLOCK_Y):
        offsets_y = pid_y * BLOCK_X * BLOCK_Y + \
            BLOCK_X * ydim + tl.arange(0, BLOCK_X)
        # for j in range(out_feat_len):
        for j in range(pid_z * BLOCK_Z, (pid_z + 1) * BLOCK_Z):
            accumulator = tl.zeros([BLOCK_X], dtype=tl.float32)
            for k in range(neighbor_start, neighbor_end):
                neighbor_id = tl.load(idx + k)
                rel_id = tl.load(rel + k)
                x = tl.load(input + neighbor_id * feat_len + offsets_y)
                offset_weight = rel_id * out_feat_len * feat_len + feat_len * j + pid_y * \
                    BLOCK_X * BLOCK_Y + BLOCK_X * ydim + tl.arange(0, BLOCK_X)
                weight_ptr = weights + offset_weight
                w = tl.load(weight_ptr)
                accumulator += x * w
            tl.atomic_add(output + pid * out_feat_len + j + offset_output,
                          tl.sum(accumulator, axis=0))


def rgcn_triton(x: torch.Tensor, ptr: torch.Tensor, idx: torch.Tensor,
                rel: torch.Tensor, weights: torch.Tensor, num_nodes: int):
    output = torch.zeros((num_nodes, weights.shape[-1]),
                         dtype=torch.float32,
                         device=x.device)
    assert x.is_cuda and ptr.is_cuda and idx.is_cuda and rel.is_cuda and weights.is_cuda
    feat_len = x.shape[1]
    out_feat_len = weights.shape[-1]

    def grid(meta):
        return (num_nodes,
                triton.cdiv(feat_len, meta['BLOCK_X'] * meta['BLOCK_Y']),
                triton.cdiv(out_feat_len, meta['BLOCK_Z']))

    rgcn_kernel[grid](x,
                      ptr,
                      idx,
                      rel,
                      weights,
                      output,
                      feat_len,
                      out_feat_len=out_feat_len)
    return output


def rgcn_scatter(x: torch.Tensor, ptr: torch.Tensor, idx: torch.Tensor,
                 rel: torch.Tensor, weights: torch.Tensor, num_nodes: int):
    scatter_weight = weights[rel]
    mid = torch.mm(x[idx], scatter_weight)
    output = torch.zeros((num_nodes, weights.shape[-1]),
                         dtype=torch.float32,
                         device=x.device)
    deg = ptr[1:] - ptr[:-1]  # in degree
    target = torch.repeat_interleave(torch.arange(num_nodes, device=x.device),
                                     deg)
    output.index_add_(0, target, mid)
    return output


def rgcn_full_mm(x: torch.Tensor, ptr: torch.Tensor, idx: torch.Tensor,
                 rel: torch.Tensor, weights: torch.Tensor, num_nodes: int,
                 num_rel: int):
    output = torch.zeros((num_nodes, weights.shape[-1]),
                         dtype=torch.float32,
                         device=x.device)
    for i in range(num_rel):
        mm_output = torch.mm(x, weights[i])
        cxgnncomp_backend.selective_aggr(mm_output, ptr, idx, rel == i, output)
    return output