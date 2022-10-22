import torch
import cxgnncomp as cxgc
from cxgnncomp.codegen.util import compare, prof


def prepare_data():
    torch.manual_seed(0)
    # dataset_name = "paper100m"
    # file_dir = "/home/huangkz/repos/new-diskgnn/DiskGNN/graph_loader/{}_batch.pt".format(
    #     dataset_name)
    file_dir = "/home/huangkz/repos/CxGNN-DL/dump.pt"
    batch = torch.load(file_dir)
    feat_len = 128
    x = torch.randn([batch["num_node_in_layer"][-1], feat_len],
                    dtype=torch.float32,
                    device='cuda')
    ptr = batch["ptr"].cuda()
    idx = batch["idx"].cuda()
    return x, ptr, idx, batch


if __name__ == "__main__":
    num_rel = 7
    x, ptr, idx, batch = prepare_data()
    num_center = ptr.shape[0] - 1
    rels = torch.randint(0,
                         num_rel, [idx.shape[0]],
                         dtype=torch.int32,
                         device='cuda')
    weights = torch.randn([num_rel, 128, 128],
                          dtype=torch.float32,
                          device='cuda')
    transposed_weights = weights.transpose(1, 2).contiguous()
    print(x.shape, transposed_weights.shape)
    # print(batch.num_node_in_layer, batch.num_edge_in_layer)
    output1 = cxgc.rgcn_triton(x, ptr, idx, rels, weights, num_center)
    # output2 = cxgc.rgcn_scatter(x, ptr, idx, rels, weights, num_center)
    output2 = cxgc.aggr_rel_direct(x, ptr, idx, transposed_weights, rels,
                                   num_center, num_rel)
    output3 = cxgc.rgcn_full_mm(x, ptr, idx, rels, weights, num_center,
                                num_rel)
    compare(output1, output2)
    compare(output1, output3)
    prof("rgcn", "triton",
         lambda: cxgc.rgcn_triton(x, ptr, idx, rels, weights, num_center))
    prof(
        "rgcn", "manual", lambda: cxgc.aggr_rel_direct(
            x, ptr, idx, transposed_weights, rels, num_center, num_rel))
    prof(
        "rgcn", "full_mm", lambda: cxgc.rgcn_full_mm(
            x, ptr, idx, rels, weights, num_center, num_rel))
