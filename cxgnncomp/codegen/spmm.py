import triton
import triton.language as tl
import torch


@triton.autotune(configs=[
    triton.Config({'BLOCK_X': 16, 'BLOCK_Y': 1}),
    triton.Config({'BLOCK_X': 32, 'BLOCK_Y': 1}),
    triton.Config({'BLOCK_X': 64, 'BLOCK_Y': 1}),
    triton.Config({'BLOCK_X': 128, 'BLOCK_Y': 1}),
    triton.Config({'BLOCK_X': 256, 'BLOCK_Y': 1}),
    triton.Config({'BLOCK_X': 512, 'BLOCK_Y': 1}),
    triton.Config({'BLOCK_X': 16, 'BLOCK_Y': 2}),
    triton.Config({'BLOCK_X': 32, 'BLOCK_Y': 2}),
    triton.Config({'BLOCK_X': 64, 'BLOCK_Y': 2}),
    triton.Config({'BLOCK_X': 128, 'BLOCK_Y': 2}),
    triton.Config({'BLOCK_X': 256, 'BLOCK_Y': 2}),
    triton.Config({'BLOCK_X': 512, 'BLOCK_Y': 2})
], key=["feat_len"])
@triton.jit
def spmm_kernel(
    input,
    ptr,
    idx,
    output,
    feat_len,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # * BLOCK_X // 32
    pid_y = tl.program_id(axis=1)
    block_start = tl.load(ptr + pid)
    block_end = tl.load(ptr + pid + 1)
    for ydim in range(BLOCK_Y):
        offsets_y = pid_y * BLOCK_X * BLOCK_Y + \
            BLOCK_X * ydim + tl.arange(0, BLOCK_X)
        mask = offsets_y < feat_len
        accumulator = tl.zeros([BLOCK_X], dtype=tl.float32)
        for k in range(block_start, block_end):
            neighbor_id = tl.load(idx + k)
            x = tl.load(input + neighbor_id * feat_len + offsets_y, mask=mask)
            accumulator += x
        tl.store(output + pid * feat_len + offsets_y, accumulator, mask=mask)

@triton.autotune(configs=[
    triton.Config({'BLOCK_X': 16, 'BLOCK_Y': 1}),
    triton.Config({'BLOCK_X': 32, 'BLOCK_Y': 1}),
    triton.Config({'BLOCK_X': 64, 'BLOCK_Y': 1}),
    triton.Config({'BLOCK_X': 128, 'BLOCK_Y': 1}),
    triton.Config({'BLOCK_X': 256, 'BLOCK_Y': 1}),
    triton.Config({'BLOCK_X': 512, 'BLOCK_Y': 1}),
    triton.Config({'BLOCK_X': 16, 'BLOCK_Y': 2}),
    triton.Config({'BLOCK_X': 32, 'BLOCK_Y': 2}),
    triton.Config({'BLOCK_X': 64, 'BLOCK_Y': 2}),
    triton.Config({'BLOCK_X': 128, 'BLOCK_Y': 2}),
    triton.Config({'BLOCK_X': 256, 'BLOCK_Y': 2}),
    triton.Config({'BLOCK_X': 512, 'BLOCK_Y': 2})
], key=["feat_len"])
@triton.jit
def spmm_with_value_kernel(
    input,
    ptr,
    idx,
    val,
    output,
    feat_len,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # * BLOCK_X // 32
    pid_y = tl.program_id(axis=1)
    block_start = tl.load(ptr + pid)
    block_end = tl.load(ptr + pid + 1)
    for ydim in range(BLOCK_Y):
        offsets_y = pid_y * BLOCK_X * BLOCK_Y + \
            BLOCK_X * ydim + tl.arange(0, BLOCK_X)
        mask = offsets_y < feat_len
        accumulator = tl.zeros([BLOCK_X], dtype=tl.float32)
        for k in range(block_start, block_end):
            neighbor_id = tl.load(idx + k)
            value = tl.load(val + k)
            x = tl.load(input + neighbor_id * feat_len + offsets_y, mask=mask)
            accumulator += x * value
        tl.store(output + pid * feat_len + offsets_y, accumulator, mask=mask)

@triton.autotune(configs=[
    triton.Config({'BLOCK_X': 16, 'BLOCK_Y': 1}),
    triton.Config({'BLOCK_X': 32, 'BLOCK_Y': 1}),
    triton.Config({'BLOCK_X': 64, 'BLOCK_Y': 1}),
    triton.Config({'BLOCK_X': 128, 'BLOCK_Y': 1}),
    triton.Config({'BLOCK_X': 256, 'BLOCK_Y': 1}),
    # triton.Config({'BLOCK_X': 512, 'BLOCK_Y': 1}),
    triton.Config({'BLOCK_X': 16, 'BLOCK_Y': 2}),
    triton.Config({'BLOCK_X': 32, 'BLOCK_Y': 2}),
    triton.Config({'BLOCK_X': 64, 'BLOCK_Y': 2}),
    triton.Config({'BLOCK_X': 128, 'BLOCK_Y': 2}),
    # triton.Config({'BLOCK_X': 256, 'BLOCK_Y': 2}),
    # triton.Config({'BLOCK_X': 512, 'BLOCK_Y': 2})
], key=["feat_len"])
@triton.jit
def spmm_mm_kernel(
    input,
    ptr,
    idx,
    weight,
    output,
    feat_len,
    output_feat_len,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # * BLOCK_X // 32
    pid_y = tl.program_id(axis=1)
    block_start = tl.load(ptr + pid)
    block_end = tl.load(ptr + pid + 1)
    for ydim in range(BLOCK_Y):
        offsets_y = pid_y * BLOCK_X * BLOCK_Y + BLOCK_X * ydim + tl.arange(0, BLOCK_X)
        mask = offsets_y < feat_len
        accumulator = tl.zeros([BLOCK_X], dtype=tl.float32)
        for k in range(block_start, block_end):
            neighbor_id = tl.load(idx + k)
            x = tl.load(input + neighbor_id * feat_len + offsets_y, mask=mask)
            accumulator += x
        # tl.atomic_add(output + pid * output_feat_len + offsets_y, accumulator, mask=mask)
        # tl.atomic_add(output + pid * feat_len + offsets_y, tl.min(accumulator, axis=0))
        # tl.store(output + pid * feat_len + offsets_y, accumulator, mask=mask)
        for i in range(0, output_feat_len):
            w = tl.load(weight + i * feat_len + offsets_y, mask=mask)
            tl.store(output + pid * output_feat_len + i, tl.sum(accumulator*w, axis=0))
            # tl.atomic_add(output + pid * output_feat_len + i, tl.sum(accumulator*w, axis=0))


def spmm_triton(x: torch.Tensor, ptr: torch.Tensor, idx: torch.Tensor, num_nodes: int):
    output = torch.empty(
        (num_nodes, x.shape[1]), dtype=torch.float32, device=x.device)
    assert x.is_cuda and ptr.is_cuda and idx.is_cuda 
    feat_len = x.shape[1]

    def grid(meta): return (num_nodes, triton.cdiv(
        feat_len, meta['BLOCK_X'] * meta['BLOCK_Y']))
    spmm_kernel[grid](x, ptr, idx, output, feat_len)
    return output

def spmm_with_value_triton(x: torch.Tensor, ptr: torch.Tensor, idx: torch.Tensor, val: torch.Tensor, num_nodes: int):
    output = torch.empty(
        (num_nodes, x.shape[1]), dtype=torch.float32, device=x.device)
    assert x.is_cuda and ptr.is_cuda and idx.is_cuda 
    feat_len = x.shape[1]

    def grid(meta): return (num_nodes, triton.cdiv(
        feat_len, meta['BLOCK_X'] * meta['BLOCK_Y']))
    spmm_with_value_kernel[grid](x, ptr, idx, val, output, feat_len)
    return output


def spmm_mm_triton(x: torch.Tensor, ptr: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor, num_nodes: int):
    feat_len = x.shape[1]
    output_feat_len = weight.shape[0] # weight is transposed
    output = torch.zeros(
        (num_nodes, output_feat_len), dtype=torch.float32, device=x.device)
    def grid(meta): return (num_nodes, triton.cdiv(
        feat_len, meta['BLOCK_X'] * meta['BLOCK_Y']))
    spmm_mm_kernel[grid](x, ptr, idx, weight, output, feat_len, output_feat_len)
    return output

# def test_aggr():
#     task_name = "aggregation"
#     val = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                        device='cuda', dtype=torch.float32)
#     x, ptr, idx, b = prepare_data()
#     output1 = spmm(x, ptr, idx, val, b["num_node_in_layer"][-2])
#     output2 = graph_loader_backend.sage_sum_forward(
#         x, ptr, idx, b["num_node_in_layer"][-2])
#     compare(output1, output2)
#     prof(task_name, "triton", lambda: spmm(
#         x, ptr, idx, val, b["num_node_in_layer"][-2]))
#     prof(task_name, "manual", lambda: graph_loader_backend.sage_sum_forward(
#         x, ptr, idx, b["num_node_in_layer"][-2]))
