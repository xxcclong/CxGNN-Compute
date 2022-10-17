import cxgnncomp as cxgc
import torch
from cxgnncomp.codegen.spmm import spmm_triton

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
    val = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       device='cuda',
                       dtype=torch.float32)
    num_center = b["num_node_in_layer"][-2].item()
    output1 = cxgc.sage_sum_forward(x, ptr, idx, num_center)
    output2 = cxgc.spmm_triton(x, ptr, idx, val, b["num_node_in_layer"][-2])
    compare(output1, output2)
    cxgc.prof(
        "spmm", "triton",
        lambda: cxgc.spmm_triton(x, ptr, idx, val, b["num_node_in_layer"][-2]))
    cxgc.prof("spmm", "manual",
              lambda: cxgc.sage_sum_forward(x, ptr, idx, num_center))
