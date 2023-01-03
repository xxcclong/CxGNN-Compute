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


# @triton.autotune(configs=config_arr, key=["feat_len"])
# @triton.jit
# def rgcn_kernel_opt(
#     input,
#     ptr,
#     idx,
#     rel,
#     weights,
#     output,
#     feat_len,
#     out_feat_len: tl.constexpr,
#     BLOCK_X: tl.constexpr,
#     BLOCK_Y: tl.constexpr,
#     BLOCK_Z: tl.constexpr,
# ):
#     pid_x = tl.program_id(axis=0)  # * BLOCK_X // 32
#     pid_y = tl.program_id(axis=1)
#     neighbor_start = tl.load(ptr + pid_x)
#     neighbor_end = tl.load(ptr + pid_x + 1)
#     offset_output = tl.arange(0, 1)

#     for ydim in range(BLOCK_Y):
#         offsets_y = pid_y * BLOCK_X * BLOCK_Y + \
#             BLOCK_X * ydim + tl.arange(0, BLOCK_X)
#         # for j in range(out_feat_len):
#         for j in range(pid_y * BLOCK_Z, (pid_y + 1) * BLOCK_Z):
#             accumulator = tl.zeros([BLOCK_X], dtype=tl.float32)
#             for k in range(neighbor_start, neighbor_end, 16):
#                 neighbor_id = tl.load(idx + k + tl.arange(0, 16))
#                 rel_id = tl.load(rel + k)
#                 # x = tl.load(input + neighbor_id * feat_len + offsets_y)
#                 x_ptrs = input + (neighbor_id[:, None] * feat_len +
#                                   offsets_y[None, :])
#                 x = tl.load(x_ptrs)
#                 offset_weight = rel_id * out_feat_len * feat_len + feat_len * j + pid_y * \
#                     BLOCK_X * BLOCK_Y + BLOCK_X * ydim + tl.arange(0, BLOCK_X)
#                 weight_ptr = weights + offset_weight
#                 weight_ptr = weights + pid_y
#                 w = tl.load(weight_ptr)
#                 accumulator += x * w
#             tl.atomic_add(output + pid_x * out_feat_len + j + offset_output,
#                           tl.sum(accumulator, axis=0))


@triton.autotune(
    configs=[

        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 128,
        #         'BLOCK_SIZE_N': 256,
        #         'BLOCK_SIZE_K': 32,
        #     },
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 128,
        #         'BLOCK_SIZE_N': 256,
        #         'BLOCK_SIZE_K': 32,
        #     },
        #     num_stages=3,
        #     num_warps=8),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 256,
        #         'BLOCK_SIZE_N': 128,
        #         'BLOCK_SIZE_K': 32,
        #     },
        #     num_stages=3,
        #     num_warps=8),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 256,
        #         'BLOCK_SIZE_N': 64,
        #         'BLOCK_SIZE_K': 32,
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 64,
        #         'BLOCK_SIZE_N': 256,
        #         'BLOCK_SIZE_K': 32,
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 128,
        #         'BLOCK_SIZE_N': 128,
        #         'BLOCK_SIZE_K': 32,
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 128,
        #         'BLOCK_SIZE_N': 64,
        #         'BLOCK_SIZE_K': 32,
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 64,
        #         'BLOCK_SIZE_N': 128,
        #         'BLOCK_SIZE_K': 32,
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 128,
        #         'BLOCK_SIZE_N': 32,
        #         'BLOCK_SIZE_K': 32,
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 64,
        #         'BLOCK_SIZE_N': 32,
        #         'BLOCK_SIZE_K': 32,
        #     },
        #     num_stages=5,
        #     num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
            }, ),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 16,
        #         'BLOCK_SIZE_N': 32,
        #         'BLOCK_SIZE_K': 128,
        #     },
        #     num_stages=2,
        #     num_warps=1),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def rgcn_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ptr,
    idx,
    rel,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    neighbor_start = tl.load(ptr + pid_m)
    neighbor_end = tl.load(ptr + pid_m + 1)

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                      offs_bn[None, :] * stride_bn)
    b_ptrs_run = b_ptrs
    neighbor_range = tl.arange(0, BLOCK_SIZE_M)
    neighbor_offset = idx + neighbor_range

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # for neighbors in range(0, 128, BLOCK_SIZE_M):
    # for neighbors in range(4):
    # for neighbors in range(neighbor_start, neighbor_end, BLOCK_SIZE_M):
    # for i in range(4):
    # for k in range(0, K, BLOCK_SIZE_K):
    #     for i in range(4):
    #         # a = tl.load(a_ptrs, mask=a_mask)
    #         b = tl.load(b_ptrs_run)
    #         accumulator += b
    #         # accumulator += tl.dot(a, b)
    #     b_ptrs_run += BLOCK_SIZE_K * stride_bk

    # FIXME: reordering the k loop and the neighbor loop results in bad number from b
    for k in range(0, K, BLOCK_SIZE_K):
        for neighbors in range(neighbor_start, neighbor_end, BLOCK_SIZE_M):
            neighbor_new_offset = neighbor_offset + neighbors
            neighbor_mask = (neighbor_range < neighbor_end - neighbors)
            neighbor_ids = tl.load(neighbor_new_offset, mask=neighbor_mask)
            a_ptrs = a_ptr + (neighbor_ids[:, None] * stride_am +
                              offs_k[None, :] * stride_ak)
            a_mask = ((neighbor_range[:, None]) < neighbor_end - neighbors)
            # a = tl.load(a_ptrs)
            a = tl.load(a_ptrs, mask=a_mask)
            b = tl.load(b_ptrs_run)
            # accumulator = a
            # accumulator += tl.dot(a, a)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs_run += BLOCK_SIZE_K * stride_bk

    c = tl.sum(accumulator, axis=0)
    offs_cm = pid_m
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm + stride_cn * offs_cn  # 1-dim
    tl.store(c_ptrs, c)

    # offs_cm = tl.arange(0, BLOCK_SIZE_M)[:, None] + pid_m
    # offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    # c_ptrs = c_ptr + stride_cm * offs_cm + stride_cn * offs_cn
    # tl.store(c_ptrs, accumulator)


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


def rgcn_triton_opt(x: torch.Tensor, ptr: torch.Tensor, idx: torch.Tensor,
                    rel: torch.Tensor, weights: torch.Tensor, num_nodes: int):
    print("debug here")
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

    grid = lambda META: (num_nodes * triton.cdiv(out_feat_len, META[
        'BLOCK_SIZE_N']), )

    print(x, weights, output, num_nodes, out_feat_len, feat_len, x.stride(0),
          x.stride(1), weights.stride(1), weights.stride(2), output.stride(0),
          output.stride(1), ptr, idx, rel)
    print(x.shape, weights.shape, output.shape)
    weights = weights.squeeze()

    # torch.cuda.synchronize()
    bin = rgcn_matmul_kernel[grid](
        x,
        weights,
        output,
        num_nodes,
        out_feat_len,
        feat_len,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        output.stride(0),
        output.stride(1),
        ptr,
        idx,
        rel,
    )
    torch.cuda.synchronize()
    # with open("output1.txt", 'a') as f:
    #     f.write(bin.asm["ptx"])
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


def rgcn_just_aggr(x: torch.Tensor, ptr: torch.Tensor, idx: torch.Tensor,
                   rel: torch.Tensor, weights: torch.Tensor, num_nodes: int,
                   num_rel: int):
    output = torch.zeros((num_nodes, weights.shape[-1]),
                         dtype=torch.float32,
                         device=x.device)
    for i in range(num_rel):
        cxgnncomp_backend.selective_aggr(x, ptr, idx, (rel == i), output,
                                         num_nodes)
    return output


def rgcn_just_aggr_prune(x: torch.Tensor, meta: list, num_nodes: int,
                         num_rel: int):
    output = torch.zeros((num_nodes, x.shape[1]),
                         dtype=torch.float32,
                         device=x.device)
    for i in range(num_rel):
        cxgnncomp_backend.target_aggr(
            x,
            meta[3 * i],
            meta[3 * i + 1],
            meta[3 * i + 2],
            output,
            meta[3 * i + 2].shape[0],
        )
    return output


def rgcn_full_mm(x: torch.Tensor, ptr: torch.Tensor, idx: torch.Tensor,
                 rel: torch.Tensor, weights: torch.Tensor, num_nodes: int,
                 num_rel: int):
    output = torch.zeros((num_nodes, weights.shape[-1]),
                         dtype=torch.float32,
                         device=x.device)
    for i in range(num_rel):
        mm_output = torch.mm(x, weights[i])
        cxgnncomp_backend.selective_aggr(mm_output, ptr, idx, (rel == i),
                                         output, num_nodes)
    return output


def rgcn_full_mm2(x: torch.Tensor, ptr: torch.Tensor, idx: torch.Tensor,
                  rel: torch.Tensor, weights: torch.Tensor, num_nodes: int,
                  num_rel: int):
    output = torch.zeros((num_nodes, weights.shape[-1]),
                         dtype=torch.float32,
                         device=x.device)
    for i in range(num_rel):
        aggr_output = torch.zeros((num_nodes, weights.shape[-2]),
                                  dtype=torch.float32,
                                  device=x.device)
        cxgnncomp_backend.selective_aggr(x, ptr, idx, (rel == i), aggr_output,
                                         num_nodes)
        output += torch.mm(aggr_output, weights[i])
    return output


def rgcn_prune_mm(x: torch.Tensor, weights: torch.Tensor, meta: list,
                  num_nodes: int, num_rel: int):
    outputs = []
    for i in range(num_rel):
        outputs.append(
            cxgnncomp_backend.sage_sum_forward(x, meta[3 * i], meta[3 * i + 1],
                                               0))
    mm_outputs = []
    for it, item in enumerate(outputs):
        mm_outputs.append(torch.mm(item, weights[it]))
    comp_output = torch.zeros((num_nodes, weights.shape[-1]),
                              dtype=torch.float32,
                              device=x.device)
    for i in range(num_rel):
        comp_output.index_add_(0, meta[3 * i + 2], mm_outputs[i])
    return comp_output


def rgcn_full_bmm(x: torch.Tensor, ptr: torch.Tensor, idx: torch.Tensor,
                  rel: torch.Tensor, weights: torch.Tensor, comp: torch.Tensor,
                  num_nodes: int, num_rel: int):
    output = torch.zeros((num_nodes, weights.shape[-1]),
                         dtype=torch.float32,
                         device=x.device)
    tmp = torch.mm(x, weights.view(weights.shape[0], -1)).reshape(
        -1,
        weights.shape[1],
    ).contiguous()
    for i in range(num_rel):
        # mm_output = torch.einsum("b,njb->nj", comp[0], tmp)
        mm_output = torch.mm(tmp, comp[i].view(-1,
                                               1)).view(-1, weights.shape[2])
        cxgnncomp_backend.selective_aggr(mm_output, ptr, idx, (rel == i),
                                         output, num_nodes)
    return output
