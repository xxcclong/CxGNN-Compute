import torch

import cxgnncomp as cxgc
from cxgnncomp.codegen.util import compare


def prepare_data():
    torch.manual_seed(0)
    dataset_name = "paper100m"
    file_dir = "/home/huangkz/repos/new-diskgnn/DiskGNN/graph_loader/{}_batch.pt".format(
        dataset_name)
    batch = torch.load(file_dir)
    feat_len = 256
    x = torch.randn([batch["num_node_in_layer"][-1], feat_len],
                    dtype=torch.float32,
                    device='cuda')
    ptr = batch["ptr"].cuda()
    idx = batch["idx"].cuda()
    return x, ptr, idx, batch


if __name__ == "__main__":
    x, ptr, idx, b = prepare_data()
    val = torch.randn(idx.shape, dtype=torch.float32, device='cuda')
    num_center = b["num_node_in_layer"][-2].item()
    output1 = cxgc.sage_sum_forward(x, ptr, idx, num_center)
    output2 = cxgc.spmm_triton(x, ptr, idx, num_center)
    compare(output1, output2)
    cxgc.prof("spmm", "triton",
              lambda: cxgc.spmm_triton(x, ptr, idx, num_center))
    cxgc.prof("spmm", "manual",
              lambda: cxgc.sage_sum_forward(x, ptr, idx, num_center))

    output1 = cxgc.sage_sum_forward_edge_value(x, ptr, idx, val, num_center)
    output2 = cxgc.spmm_with_value_triton(x, ptr, idx, val, num_center)
    compare(output1, output2)
    cxgc.prof(
        "spmm_with_value", "triton",
        lambda: cxgc.spmm_with_value_triton(x, ptr, idx, val, num_center))
    cxgc.prof(
        "spmm_with_value", "manual",
        lambda: cxgc.sage_sum_forward_edge_value(x, ptr, idx, val, num_center))
