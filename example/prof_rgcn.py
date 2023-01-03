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


def prepare_data_toy():
    torch.manual_seed(0)
    # x = torch.randn([128, 128], dtype=torch.float16, device='cuda')
    # x = x.to(torch.float32)
    x = torch.ones([128, 128], dtype=torch.float32, device='cuda')
    # for i in range(x.shape[0]):
    #     x[:][i] = i + 1
    # x = torch.arange(1, 128 * 128 + 1, dtype=torch.float32, device='cuda').view(128, 128) / 1000
    num_node = 31
    ptr = torch.tensor([0, num_node], dtype=torch.int64, device='cuda')
    idx = torch.arange(0, num_node, dtype=torch.int64, device='cuda')
    print(idx)
    batch = {}
    batch["num_node_in_layer"] = [128, 128]
    return x, ptr, idx, batch


if __name__ == "__main__":
    num_rel = 1
    # x, ptr, idx, batch = prepare_data()
    x, ptr, idx, batch = prepare_data_toy()
    num_center = ptr.shape[0] - 1
    rels = torch.randint(0,
                         num_rel, [idx.shape[0]],
                         dtype=torch.int32,
                         device='cuda')
    # weights = torch.arange(1, 128 * 128 + 1, dtype=torch.float32, device='cuda').view(num_rel, 128, 128)
    weights = torch.ones([num_rel, 128, 128],
                          dtype=torch.float32,
                          device='cuda')
    # for i in range(weights.shape[1]):
    #     weights[0][i] *= (i + 1)
    # weights[0][:][i] *= (i + 1)
    transposed_weights = weights.transpose(1, 2).contiguous()
    print(x.shape, transposed_weights.shape)
    # print(batch.num_node_in_layer, batch.num_edge_in_layer)
    output1 = cxgc.rgcn_triton(x, ptr, idx, rels, transposed_weights,
                               num_center)
    # output2 = cxgc.rgcn_scatter(x, ptr, idx, rels, weights, num_center)
    # output2 = cxgc.aggr_rel_direct(x, ptr, idx, weights, rels, num_center,
    #                                num_rel)
    output3 = cxgc.rgcn_full_mm(x, ptr, idx, rels, weights, num_center,
                                num_rel)
    output4 = cxgc.rgcn_full_mm2(x, ptr, idx, rels, weights, num_center,
                                 num_rel)
    # output_triton_mm_T = cxgc.rgcn_triton_opt(x, ptr, idx, rels,
    #                                           transposed_weights, num_center)
    output_triton_mm = cxgc.rgcn_triton_opt(x, ptr, idx, rels, weights,
                                            num_center)
    # compare(output1, output2)
    compare(output1, output3)
    compare(output1, output4)
    if num_rel == 1:
        # compare(output1, output_triton_mm_T)
        compare(output1, output_triton_mm)
        # torch.set_printoptions(profile="full")
        print(output1)
        print(output_triton_mm)
        print(output_triton_mm.shape)
        exit()
    prof(
        "rgcn", "triton", lambda: cxgc.rgcn_triton(
            x, ptr, idx, rels, transposed_weights, num_center))
    prof(
        "rgcn", "triton_opt", lambda: cxgc.rgcn_triton_opt(
            x, ptr, idx, rels, transposed_weights, num_center))
    prof(
        "rgcn", "manual", lambda: cxgc.aggr_rel_direct(
            x, ptr, idx, weights, rels, num_center, num_rel))
    prof(
        "rgcn", "full_mm", lambda: cxgc.rgcn_full_mm(
            x, ptr, idx, rels, weights, num_center, num_rel))
    prof(
        "rgcn", "full_mm2", lambda: cxgc.rgcn_full_mm2(
            x, ptr, idx, rels, weights, num_center, num_rel))
    prof(
        "rgcn", "just aggr", lambda: cxgc.rgcn_just_aggr(
            x, ptr, idx, rels, weights, num_center, num_rel))

    # num_base = 2
    # bmm_weights = torch.randn([128, num_base, 128],
    #                           dtype=torch.float32,
    #                           device='cuda')
    # comp = torch.randn([num_rel, num_base], dtype=torch.float32, device='cuda')
    # prof(
    #     "rgcn", "bmm", lambda: cxgc.rgcn_full_bmm(
    #         x, ptr, idx, rels, bmm_weights, comp, num_center, num_rel))

    meta = cxgc.rel_schedule(ptr.cpu(), idx.cpu(), rels.cpu(),
                             batch["num_node_in_layer"], num_rel)
    for i in range(0, len(meta) - 1):
        meta[i] = meta[i].cuda()
    comp_output1 = cxgc.rgcn_just_aggr_prune(x, meta, num_center, num_rel)
    comp_output2 = cxgc.rgcn_just_aggr(x, ptr, idx, rels, weights, num_center,
                                       num_rel)
    compare(comp_output1, comp_output2)
    prof("rgcn", "just aggr prune",
         lambda: cxgc.rgcn_just_aggr_prune(x, meta, num_center, num_rel))

    cxgc.prof("spmm", "manual",
              lambda: cxgc.sage_sum_forward(x, ptr, idx, num_center))

    comp_output3 = cxgc.rgcn_prune_mm(x, weights, meta, num_center, num_rel)
    compare(output1, comp_output3)

    prof("rgcn", "full prune",
         lambda: cxgc.rgcn_prune_mm(x, weights, meta, num_center, num_rel))
