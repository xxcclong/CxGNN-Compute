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


def prepare_graph(dset="products",
                  feat_len=128,
                  num_head=1,
                  need_edge_index=False,
                  need_feat=True,
                  undirected=True,
                  num_seeds=1000,
                  is_full_graph=True):
    if is_full_graph:
        return prepare_data_full_graph(dset=dset,
                                       feat_len=feat_len,
                                       num_head=num_head,
                                       need_edge_index=need_edge_index,
                                       need_feat=need_feat,
                                       undirected=undirected)
    else:
        return prepare_data_sampled_graph(dset=dset,
                                          feat_len=feat_len,
                                          num_head=num_head,
                                          need_edge_index=need_edge_index,
                                          num_seeds=num_seeds)


def prepare_data_full_graph(
    dset="products",
    feat_len=128,
    num_head=1,
    need_edge_index=False,
    need_feat=True,
    undirected=True,
):
    print(
        f"=======\nLoading full graph structure... dataset={dset} feature length={feat_len * need_feat} num_head={num_head} undirected={undirected}\n======="
    )
    ptr, idx = cxgnndl.load_full_graph_structure(dset, undirected)
    ptr = torch.from_numpy(ptr).cuda()
    idx = torch.from_numpy(idx).cuda()
    num_node = max(torch.max(idx) + 1, ptr.shape[0] - 1)
    if feat_len == 0:
        need_feat = False
    if ptr.shape[0] - 1 != num_node:
        new_ptr = torch.zeros([num_node + 1], dtype=torch.int64, device="cuda")
        new_ptr[:ptr.shape[0]] = ptr
        new_ptr[ptr.shape[0]:] = ptr[-1]
        ptr = new_ptr
    if need_feat:
        if num_head == 1:
            x = torch.randn([ptr.shape[0] - 1, feat_len],
                            dtype=torch.float32,
                            device='cuda')
        else:
            x = torch.randn([ptr.shape[0] - 1, num_head, feat_len],
                            dtype=torch.float32,
                            device='cuda')
    else:
        x = None
    batch = {}
    batch["num_node_in_layer"] = torch.tensor([ptr.shape[0] - 1] * 4)
    batch["num_edge_in_layer"] = torch.tensor([idx.shape[0]] * 4)
    print("After loading full graph structure...")
    print(f"num_edge {idx.shape[0]} num_center {ptr.shape[0] - 1}")
    print(f"num_node_in_layer {batch['num_node_in_layer']}")
    if need_edge_index:
        edge_index = torch.stack([
            idx,
            torch.repeat_interleave(torch.arange(
                0, ptr.shape[0] - 1, device="cuda"),
                                    repeats=ptr[1:] - ptr[:-1])
        ],
                                 dim=0)
        return x, ptr, idx, batch, edge_index
    else:
        return x, ptr, idx, batch


def prepare_data_sampled_graph(
    dset,
    num_seeds,
    feat_len=128,
    num_head=1,
    fanouts=[10, 15, 20],
    need_edge_index=False,
):
    full_ptr, full_idx = cxgnndl.load_full_graph_structure(dset)
    full_ptr = torch.from_numpy(full_ptr)
    full_idx = torch.from_numpy(full_idx)
    num_all_nodes = full_ptr.shape[0] - 1
    seed_nodes = torch.randint(0, num_all_nodes, (num_seeds, ))
    ptr, idx, input_nodes, num_node_in_layer, num_edge_in_layer = cxgnndl_backend.neighbor_sample(
        full_ptr, full_idx, fanouts, seed_nodes)
    ptr = ptr.cuda()
    idx = idx.cuda()
    if num_head == 1:
        x = torch.randn([input_nodes.shape[0], feat_len],
                        dtype=torch.float32,
                        device='cuda')
    else:
        x = torch.randn([input_nodes.shape[0], num_head, feat_len],
                        dtype=torch.float32,
                        device='cuda')
    batch = {}
    batch["num_node_in_layer"] = num_node_in_layer
    batch["num_edge_in_layer"] = num_edge_in_layer
    batch["sub_to_full"] = input_nodes.cuda()
    print(ptr.shape, idx.shape, x.shape, batch["num_node_in_layer"])
    if need_edge_index:
        edge_index = torch.stack([
            idx,
            torch.repeat_interleave(torch.arange(
                0, ptr.shape[0] - 1, device="cuda"),
                                    repeats=ptr[1:] - ptr[:-1])
        ],
                                 dim=0)
        return x, ptr, idx, batch, edge_index
    else:
        return x, ptr, idx, batch
