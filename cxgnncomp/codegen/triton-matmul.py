import torch

import triton
import triton.language as tl


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
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    index_ptr,
    rel_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
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
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse
    # See above `L2 Cache Optimizations` section for details
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
    offs_am_index = tl.load(index_ptr + offs_am)
    rel = tl.load(rel_ptr + pid_m)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_mask = (offs_am_index[:, None] >= 0) & (offs_k[None, :] < K)
    a_ptrs = a_ptr + (offs_am_index[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    b_ptr = b_ptr + rel * K * N
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                      offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=a_mask)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # if ACTIVATION == "leaky_relu":
    #     accumulator = leaky_relu(accumulator)
    # c = accumulator.to(tl.float16)

    # offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # c_ptrs = c_ptr + stride_cm * offs_cm[:,
    #                                      None] + stride_cn * offs_cn[None, :]
    c_ptrs = c_ptr + stride_cm * offs_am_index[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = (offs_am_index[:, None] >= 0) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def test_kernel(
    a_ptr,
    c_ptr,
    idx_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # if pid == 0:
    #     # offs_a = [1, 3, 5, 8]
    offs_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    idx = tl.load(idx_ptr + offs_idx)
    a = tl.load(a_ptr + idx)
    output = a + 1
    c = c_ptr + offs_idx
    tl.store(c, a)
    # c = c_ptr + offs_idx


def run_test(a: torch.Tensor, b: torch.Tensor, idx: torch.Tensor):
    # grid = lambda META: (32, 32)
    num_elem = a.shape[0]

    # def grid(meta):
    #     return (triton.cdiv(num_elem, meta["BLOCK_SIZE"]))
    grid = lambda meta: (triton.cdiv(num_elem, meta["BLOCK_SIZE"]), 1, 1)

    test_kernel[grid](a, b, idx, BLOCK_SIZE=32)


# a = torch.arange(0, 3200, dtype=torch.float32).reshape(-1).cuda()
# b = torch.zeros(100, 32, dtype=torch.float32).cuda().reshape(-1)
# # idx = torch.tensor([1, 3, 5, 8], dtype=torch.int64, device=a.device)
# idx = torch.randint(0, 3200, (3200, ), dtype=torch.int64,
#                     device=a.device).reshape(-1)
# print(idx)
# run_test(a, b, idx)
# print(b)
# exit()


# we can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a, b, idx, rel, activation=""):
    # a: [num_edge, K]
    # b: [R, K, N]
    # checks constraints
    assert a.shape[1] == b.shape[1], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    M, K = a.shape
    R, K, N = b.shape
    M = idx.shape[0]
    assert (
        K % 32 == 0
    ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(
        N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a,
        b,
        c,
        idx,
        rel,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(1),
        b.stride(2),
        # b.stride(0),
        # b.stride(1),
        c.stride(0),
        c.stride(1),
        ACTIVATION=activation,
    )
    return c


torch.manual_seed(0)
num_edge = 2332486
a = torch.randn((num_edge, 256), device='cuda', dtype=torch.float32)
b = torch.randn((256, 256), device='cuda', dtype=torch.float32)
num_rel = 7
rel = torch.randint(0, num_rel, (num_edge, ), dtype=torch.int64)

visited = torch.zeros((num_edge, ), dtype=torch.bool)
thres = 32
new_idx = []
batched_rels = []
mapping = torch.ones((num_edge, ), dtype=torch.int64) * -1
total_num = 0
for i in range(num_edge):
    if visited[i]:
        continue
    visited[i] = True
    cnt = 1
    the_rel = rel[i]
    new_idx.append(i)
    mapping[i] = total_num
    total_num += 1
    batched_rels.append(the_rel)
    for j in range(i + 1, num_edge):
        if the_rel == rel[j]:
            visited[j] = True
            cnt += 1
            new_idx.append(j)
            mapping[j] = total_num
            total_num += 1
            if cnt == thres:
                break
    if cnt < thres:
        for j in range(thres - cnt):
            new_idx.append(-1)
new_idx = torch.tensor(new_idx, dtype=torch.int64, device=a.device)
batched_rels = torch.tensor(batched_rels, dtype=torch.int64, device=a.device)
print(new_idx.shape)
print(batched_rels.shape)

# rel_b = torch.randn((num_rel, 256, 256), device='cuda', dtype=torch.float32)
rel_b = torch.repeat_interleave(b.unsqueeze(0), num_rel,
                                dim=0).reshape(num_rel, 256, 256)
triton_output = matmul(a, rel_b, new_idx, batched_rels)[:num_edge]
torch.cuda.synchronize()
torch_output = torch.matmul(a, b)
print(f"triton_output={triton_output}")
print(f"torch_output={torch_output}")
print(triton.testing.allclose(triton_output, torch_output))
# if triton.testing.allclose(triton_output, torch_output):
#     print("✅ Triton and Torch match")
# else:
#     print("❌ Triton and Torch differ")
print(triton.testing.do_bench(lambda: matmul(a, rel_b, new_idx, batched_rels)))
print(triton.testing.do_bench(lambda: matmul(a, rel_b, new_idx, batched_rels)))
print(triton.testing.do_bench(lambda: torch.matmul(a, b)))
