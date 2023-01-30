import torch, cxgnndl, cxgnndl_backend

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


def prepare_data_full_graph(dset):
    ptr, idx = cxgnndl.load_full_graph_structure(dset)
    ptr = torch.from_numpy(ptr).cuda()
    idx = torch.from_numpy(idx).cuda()
    feat_len = 128
    x = torch.randn([ptr.shape[0] - 1, feat_len],
                    dtype=torch.float32,
                    device='cuda')
    # edge_index = torch.stack([
    #     idx,
    #     torch.repeat_interleave(torch.arange(
    #         0, ptr.shape[0] - 1, device="cuda"),
    #                             repeats=ptr[1:] - ptr[:-1])
    # ],
    #                          dim=0)
    batch = {}
    batch["num_node_in_layer"] = torch.tensor(
        [ptr.shape[0] - 1, ptr.shape[0] - 1])
    return x, ptr, idx, batch


def prepare_data_sampled_graph(dset, num_seeds, fanouts=[10, 15, 20]):
    full_ptr, full_idx = cxgnndl.load_full_graph_structure(dset)
    full_ptr = torch.from_numpy(full_ptr)
    full_idx = torch.from_numpy(full_idx)
    num_all_nodes = full_ptr.shape[0] - 1
    seed_nodes = torch.randint(0, num_all_nodes, (num_seeds, ))
    ptr, idx, input_nodes, num_node_in_layer, num_edge_in_layer = cxgnndl_backend.neighbor_sample(
        full_ptr, full_idx, fanouts, seed_nodes)
    ptr = ptr.cuda()
    idx = idx.cuda()
    feat_len = 128
    x = torch.randn([input_nodes.shape[0], feat_len],
                    dtype=torch.float32,
                    device='cuda')
    batch = {}
    batch["num_node_in_layer"] = num_node_in_layer
    batch["sub_to_full"] = input_nodes.cuda()
    print(ptr.shape, idx.shape, x.shape, batch["num_node_in_layer"])
    return x, ptr, idx, batch