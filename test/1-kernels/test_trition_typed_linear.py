import torch

import triton
import triton.language as tl

from cxgnncomp.codegen.triton_typed_matmul import typed_matmul

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
triton_output = typed_matmul(a, rel_b, new_idx, new_idx,
                             batched_rels)[:num_edge]
torch.cuda.synchronize()
torch_output = torch.matmul(a, b)
print(f"triton_output={triton_output}")
print(f"torch_output={torch_output}")
print(triton.testing.allclose(triton_output, torch_output))
# if triton.testing.allclose(triton_output, torch_output):
#     print("✅ Triton and Torch match")
# else:
#     print("❌ Triton and Torch differ")
print(
    triton.testing.do_bench(
        lambda: typed_matmul(a, rel_b, new_idx, new_idx, batched_rels)))
print(
    triton.testing.do_bench(
        lambda: typed_matmul(a, rel_b, new_idx, new_idx, batched_rels)))
print(triton.testing.do_bench(lambda: torch.matmul(a, b)))