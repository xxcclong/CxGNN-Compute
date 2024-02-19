import torch

import triton
import triton.language as tl

import time


# @triton.autotune(
#     configs=[
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 128,
#                 'BLOCK_SIZE_N': 256,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=3,
#             num_warps=8),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 256,
#                 'BLOCK_SIZE_N': 128,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=3,
#             num_warps=8),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 256,
#                 'BLOCK_SIZE_N': 64,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=4,
#             num_warps=4),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 64,
#                 'BLOCK_SIZE_N': 256,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=4,
#             num_warps=4),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 128,
#                 'BLOCK_SIZE_N': 128,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=4,
#             num_warps=4),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 128,
#                 'BLOCK_SIZE_N': 64,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=4,
#             num_warps=4),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 64,
#                 'BLOCK_SIZE_N': 128,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=4,
#             num_warps=4),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 128,
#                 'BLOCK_SIZE_N': 32,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=4,
#             num_warps=4),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 64,
#                 'BLOCK_SIZE_N': 32,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=5,
#             num_warps=2),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 32,
#                 'BLOCK_SIZE_N': 64,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=5,
#             num_warps=2),
#     ],
#     key=['M', 'N', 'K'],
# )
@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def typed_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    src_index_ptr,
    dst_index_ptr,
    rel_ptr,
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

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_am_index = tl.load(src_index_ptr + offs_am, mask=offs_am < M)
    rel_pos = pid_m * BLOCK_SIZE_M
    rel = tl.load(rel_ptr + rel_pos, mask=rel_pos < M)
    # if rel >= 7:
    #     return
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_mask = (offs_am_index[:, None] >= 0) & (offs_am_index[:, None] < M) & (
        offs_k[None, :] < K) & (offs_am[:, None] < M)
    a_ptrs = a_ptr + (offs_am_index[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    b_ptr = b_ptr + rel * K * N
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

    # offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # c_ptrs = c_ptr + stride_cm * offs_cm[:,
    #                                      None] + stride_cn * offs_cn[None, :]

    offs_cm_index = tl.load(dst_index_ptr + offs_am, mask=offs_am < M)
    c_ptrs = c_ptr + stride_cm * offs_cm_index[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = (offs_cm_index[:, None] >= 0) & (offs_cm_index[:, None] < M) & (
        offs_cn[None, :] < N) & (offs_am_index[:, None] >= 0) & (
            offs_am_index[:, None] < M) & (offs_am[:, None] < M)
    tl.store(c_ptrs, accumulator, mask=c_mask)
    # tl.atomic_add(c_ptrs, accumulator, mask=c_mask)


# @triton.autotune(
#     configs=[
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 128,
#                 'BLOCK_SIZE_N': 256,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=3,
#             num_warps=8),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 256,
#                 'BLOCK_SIZE_N': 128,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=3,
#             num_warps=8),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 256,
#                 'BLOCK_SIZE_N': 64,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=4,
#             num_warps=4),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 64,
#                 'BLOCK_SIZE_N': 256,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=4,
#             num_warps=4),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 128,
#                 'BLOCK_SIZE_N': 128,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=4,
#             num_warps=4),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 128,
#                 'BLOCK_SIZE_N': 64,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=4,
#             num_warps=4),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 64,
#                 'BLOCK_SIZE_N': 128,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=4,
#             num_warps=4),
#         triton.Config( # these lead to error
#             {
#                 'BLOCK_SIZE_M': 128,
#                 'BLOCK_SIZE_N': 32,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=4,
#             num_warps=4),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 64,
#                 'BLOCK_SIZE_N': 32,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=5,
#             num_warps=2),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 32,
#                 'BLOCK_SIZE_N': 64,
#                 'BLOCK_SIZE_K': 32,
#                 'GROUP_SIZE_M': 8
#             },
#             num_stages=5,
#             num_warps=2),
#     ],
#     key=['M', 'N', 'K'],
# )
@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 512,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def typed_matmul_kernel_single_index(
    a_ptr,
    b_ptr,
    c_ptr,
    src_index_ptr,
    rel_ptr,
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

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_am_index = tl.load(src_index_ptr + offs_am, mask=offs_am < M)
    rel_pos = pid_m * BLOCK_SIZE_M
    rel = tl.load(rel_ptr + rel_pos, mask=rel_pos < M)
    # if rel >= 7:
    #     return
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_mask = (offs_am_index[:, None] >= 0) & (offs_am_index[:, None] < M) & (
        offs_k[None, :] < K) & (offs_am[:, None] < M)
    a_ptrs = a_ptr + (offs_am_index[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    b_ptr = b_ptr + rel * K * N
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

    # offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # c_ptrs = c_ptr + stride_cm * offs_cm[:,
    #                                      None] + stride_cn * offs_cn[None, :]
    # offset on m for c is the same as a's
    c_ptrs = c_ptr + stride_cm * offs_am_index[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = (offs_am_index[:, None] >= 0) & (offs_am_index[:, None] < M) & (
        offs_cn[None, :] < N) & (offs_am[:, None] < M)
    tl.store(c_ptrs, accumulator, mask=c_mask)
    # tl.atomic_add(c_ptrs, accumulator, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=5,
            num_warps=2),
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
def typed_matmul_kernel_single_index_seq_output(
    a_ptr,
    b_ptr,
    c_ptr,
    src_index_ptr,
    rel_ptr,
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

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_am_index = tl.load(src_index_ptr + offs_am, mask=offs_am < M)
    rel_pos = pid_m * BLOCK_SIZE_M
    rel = tl.load(rel_ptr + rel_pos, mask=rel_pos < M)
    # if rel >= 7:
    #     return
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_mask = (offs_am_index[:, None] >= 0) & (offs_am_index[:, None] < M) & (
        offs_k[None, :] < K) & (offs_am[:, None] < M)
    a_ptrs = a_ptr + (offs_am_index[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    b_ptr = b_ptr + rel * K * N
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
    # offset for c m is sequential
    c_ptrs = c_ptr + stride_cm * offs_am[:,
                                         None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_am[:, None] >= 0) & (offs_am[:, None] <
                                        M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
    # tl.atomic_add(c_ptrs, accumulator, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=5,
            num_warps=2),
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
def typed_matmul_kernel_no_index(
    a_ptr,
    b_ptr,
    c_ptr,
    rel_ptr,
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

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rel_pos = pid_m * BLOCK_SIZE_M
    rel = tl.load(rel_ptr + rel_pos, mask=rel_pos < M)
    # if rel >= 7:
    #     return
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_mask = (offs_k[None, :] < K) & (offs_am[:, None] < M)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    b_ptr = b_ptr + rel * K * N
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
    c_mask = (offs_cn[None, :] < N) & (offs_am[:, None] < M)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def typed_matmul(a,
                 b,
                 rel,
                 num_valid_item=-1,
                 src_idx=None,
                 dst_idx=None,
                 seq_output=False):
    # print(a.shape, b.shape, rel.shape, num_valid_item)
    # if src_idx is not None:
    #     print(src_idx.shape)
    #     print(torch.max(src_idx), torch.min(src_idx))
    # if dst_idx is not None:
    #     print(dst_idx.shape)
    #     print(torch.max(dst_idx), torch.min(dst_idx))

    assert a.shape[1] == b.shape[1], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    assert rel.is_contiguous(), "matrix rel must be contiguous"
    M, K = a.shape
    R, K, N = b.shape
    if src_idx is not None:
        assert src_idx.is_contiguous(), "matrix idx must be contiguous"
        M = src_idx.shape[0]
    if num_valid_item != -1:
        M = num_valid_item  # FIXME: some item is not computed
    # print("num valid item: ", M)
    assert (K % 32 == 0)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(
        N, META['BLOCK_SIZE_N']), )
    # torch.cuda.synchronize()
    # t0 = time.time()
    if src_idx is None:
        # print("no index")
        typed_matmul_kernel_no_index[grid](
            a,
            b,
            c,
            rel,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(1),
            b.stride(2),
            c.stride(0),
            c.stride(1),
        )
    elif src_idx is not None and dst_idx is None and not seq_output:
        # print("single index")
        typed_matmul_kernel_single_index[grid](
            a,
            b,
            c,
            src_idx,
            rel,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(1),
            b.stride(2),
            c.stride(0),
            c.stride(1),
        )
        pass
    elif src_idx is not None and dst_idx is None and seq_output:
        # print("single index seq output")
        typed_matmul_kernel_single_index_seq_output[grid](
            a,
            b,
            c,
            src_idx,
            rel,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(1),
            b.stride(2),
            c.stride(0),
            c.stride(1),
        )
    else:
        # print("typed matmul kernel")
        typed_matmul_kernel[grid](
            a,
            b,
            c,
            src_idx,
            dst_idx,
            rel,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(1),
            b.stride(2),
            c.stride(0),
            c.stride(1),
        )
    # torch.cuda.synchronize()
    # print("time: ", time.time() - t0, M, N, K)
    return c
