import torch

import triton
import triton.language as tl

import time


@triton.autotune(
    configs=[
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 128,
        #         'BLOCK_SIZE_N': 256,
        #         'BLOCK_SIZE_K': 32,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=3,
        #     num_warps=8),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 256,
        #         'BLOCK_SIZE_N': 128,
        #         'BLOCK_SIZE_K': 32,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=3,
        #     num_warps=8),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 256,
        #         'BLOCK_SIZE_N': 64,
        #         'BLOCK_SIZE_K': 32,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 64,
        #         'BLOCK_SIZE_N': 256,
        #         'BLOCK_SIZE_K': 32,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 128,
        #         'BLOCK_SIZE_N': 128,
        #         'BLOCK_SIZE_K': 32,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 128,
        #         'BLOCK_SIZE_N': 64,
        #         'BLOCK_SIZE_K': 32,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 64,
        #         'BLOCK_SIZE_N': 128,
        #         'BLOCK_SIZE_K': 32,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 128,
        #         'BLOCK_SIZE_N': 32,
        #         'BLOCK_SIZE_K': 32,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 64,
        #         'BLOCK_SIZE_N': 32,
        #         'BLOCK_SIZE_K': 32,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=5,
        #     num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=5,
            num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def sddmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    dst_index_ptr,
    src_index_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # load dst
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_am_index = tl.load(dst_index_ptr + offs_am, mask=offs_am < M)

    # load src
    src_id = tl.load(
        src_index_ptr + pid_m * BLOCK_SIZE_M
    )  # assuming src index array has same shape with dst index array
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_mask = (offs_am_index[:, None] >= 0) & (offs_am_index[:, None] < M) & (
        offs_k[None, :] < K) & (offs_am[:, None] < M)
    a_ptrs = a_ptr + (offs_am_index[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    b_ptr = b_ptr + src_id * K * N
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                      offs_bn[None, :] * stride_bn)
    b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=a_mask)
        b = tl.load(b_ptrs, mask=b_mask)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_am[:,
                                         None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_am[:, None] >= 0) & (offs_am[:, None] <
                                        M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def sddmm_dense(dst_feat_mat, src_feat_mat, dst_index, src_index, num_edge):
    assert dst_feat_mat.is_cuda and src_feat_mat.is_cuda and dst_index.is_cuda and src_index.is_cuda, f"{dst_feat_mat.is_cuda}, {src_feat_mat.is_cuda}, {dst_index.is_cuda}, {src_index.is_cuda}"
    assert dst_feat_mat.is_contiguous(), "matrix A must be contiguous"
    assert src_feat_mat.is_contiguous(), "matrix B must be contiguous"
    assert len(src_feat_mat.shape) == 3  # num_src, feat, head
    K = dst_feat_mat.shape[1]
    assert src_feat_mat.shape[1] == K
    num_head = src_feat_mat.shape[2]
    N = num_head
    M = num_edge

    output_mat = torch.empty((M, N),
                             device=dst_feat_mat.device,
                             dtype=dst_feat_mat.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(
        N, META['BLOCK_SIZE_N']), )
    sddmm_kernel[grid](
        dst_feat_mat,
        src_feat_mat,
        output_mat,
        dst_index,
        src_index,
        M,
        N,
        K,
        dst_feat_mat.stride(0),
        dst_feat_mat.stride(1),
        src_feat_mat.stride(1),
        src_feat_mat.stride(2),
        output_mat.stride(0),
        output_mat.stride(1),
    )
    return output_mat