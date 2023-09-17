import torch, cxgnndl, cxgnndl_backend
import numpy as np
import cxgnncomp_backend


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
                  need_edge_index=0,
                  undirected=True,
                  num_seeds=1000,
                  is_full_graph=1,
                  need_feat=True,
                  device="cuda",
                  rank=-1):
    if is_full_graph == 1:
        return prepare_data_full_graph(dset=dset,
                                       feat_len=feat_len,
                                       num_head=num_head,
                                       need_edge_index=need_edge_index,
                                       undirected=undirected,
                                       need_feat=need_feat,
                                       device=device)
    elif is_full_graph == 2:
        return prepare_data_full_graph_training_set(
            dset=dset,
            feat_len=feat_len,
            num_head=num_head,
            need_edge_index=need_edge_index,
            undirected=undirected,
            need_feat=need_feat,
            device=device,
            rank=rank)
    else:
        return prepare_data_sampled_graph(dset=dset,
                                          feat_len=feat_len,
                                          num_head=num_head,
                                          need_edge_index=need_edge_index,
                                          num_seeds=num_seeds,
                                          device=device)


def prepare_data_full_graph(dset="products",
                            feat_len=128,
                            num_head=1,
                            need_edge_index=0,
                            need_feat=True,
                            undirected=True,
                            device="cuda"):
    print(
        f"=======\nLoading full graph structure... dataset={dset} feature length={feat_len * need_feat} num_head={num_head} undirected={undirected}\n======="
    )
    ptr, idx = cxgnndl.load_full_graph_structure(dset, undirected)
    ptr = torch.from_numpy(ptr).to(device)
    idx = torch.from_numpy(idx).to(device)
    num_node = max(torch.max(idx) + 1, ptr.shape[0] - 1)
    if feat_len == 0:
        need_feat = False
    if ptr.shape[0] - 1 != num_node:
        new_ptr = torch.zeros([num_node + 1], dtype=torch.int64, device=device)
        new_ptr[:ptr.shape[0]] = ptr
        new_ptr[ptr.shape[0]:] = ptr[-1]
        ptr = new_ptr
    if need_feat:
        if num_head == 1:
            x = torch.randn([ptr.shape[0] - 1, feat_len],
                            dtype=torch.float32,
                            device=device)
        else:
            x = torch.randn([ptr.shape[0] - 1, num_head, feat_len],
                            dtype=torch.float32,
                            device=device)
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
                0, ptr.shape[0] - 1, device=device),
                                    repeats=ptr[1:] - ptr[:-1])
        ],
                                 dim=0)
        return x, ptr, idx, batch, edge_index
    else:
        return x, ptr, idx, batch


def prepare_data_full_graph_training_set(dset="products",
                                         feat_len=128,
                                         num_head=1,
                                         need_edge_index=0,
                                         need_feat=True,
                                         undirected=True,
                                         device="cuda",
                                         rank=-1,
                                         num_layer=3):
    print(
        f"=======\nLoading full graph structure **training set**... dataset={dset} feature length={feat_len * need_feat} num_head={num_head} undirected={undirected}\n======="
    )
    assert rank != -1, "rank must be specified"
    ptr, idx = cxgnndl.load_full_graph_structure(dset, undirected)
    ptr = torch.from_numpy(ptr).to(device)
    idx = torch.from_numpy(idx).to(device)
    num_node = max(torch.max(idx).item() + 1, ptr.shape[0] - 1)

    training_nodes = torch.from_numpy(
        np.fromfile(
            f"/home/huangkz/data/dataset_diskgnn/{dset}/processed/split/time/train_idx.dat",
            dtype=np.int64)).to(device)
    training_mask = torch.zeros([ptr.shape[0] - 1],
                                dtype=torch.float32,
                                device=device)
    training_mask[training_nodes] = 1
    training_nodes = training_mask
    num_node_in_layer = []
    num_node_in_layer.append((training_nodes.abs() > 1e-6).sum().item())
    torch.cuda.set_device(rank)
    visit_mask = torch.ones(num_node, device=device) * -1
    if device.lower() == "cpu":
        training_nodes = training_nodes.to(rank)
        ptr = ptr.to(rank)
        idx = idx.to(rank)
        visit_mask = visit_mask.to(rank)
    visit_mask[training_nodes > 0] = num_layer
    for i in range(num_layer):
        training_nodes += cxgnncomp_backend.spmv(ptr, idx, training_nodes,
                                                 ptr.shape[0] - 1)
        torch.cuda.synchronize(ptr.device)
        num_node_in_layer.append((training_nodes.abs() > 1e-6).sum().item())
        visit_mask[torch.logical_and(training_nodes > 0,
                                     visit_mask == -1)] = num_layer - i - 1
    del training_nodes
    # print(rank, num_node_in_layer)
    # for i in range(num_layer + 1):
    #     print((visit_mask >= i).sum())

    if device.lower() == "cpu":
        ptr = ptr.to(device)
        idx = idx.to(device)
        visit_mask = visit_mask.to(device)

    if feat_len == 0:
        need_feat = False
    if ptr.shape[0] - 1 != num_node:
        new_ptr = torch.zeros([num_node + 1], dtype=torch.int64, device=device)
        new_ptr[:ptr.shape[0]] = ptr
        new_ptr[ptr.shape[0]:] = ptr[-1]
        ptr = new_ptr
    if need_feat:
        if num_head == 1:
            x = torch.randn([ptr.shape[0] - 1, feat_len],
                            dtype=torch.float32,
                            device=device)
        else:
            x = torch.randn([ptr.shape[0] - 1, num_head, feat_len],
                            dtype=torch.float32,
                            device=device)
    else:
        x = None
    batch = {}
    batch["num_node_in_layer"] = num_node_in_layer
    batch["visit_mask"] = visit_mask
    batch["num_edge_in_layer"] = None
    print("After loading full graph structure...")
    print(f"num_edge {idx.shape[0]} num_center {ptr.shape[0] - 1}")
    # print(f"num_node_in_layer {batch['num_node_in_layer']}")
    if need_edge_index:
        edge_index = torch.stack([
            idx,
            torch.repeat_interleave(torch.arange(
                0, ptr.shape[0] - 1, device=device),
                                    repeats=ptr[1:] - ptr[:-1])
        ],
                                 dim=0)
        return x, ptr, idx, batch, edge_index
    else:
        return x, ptr, idx, batch


def prepare_data_sampled_graph(dset,
                               num_seeds,
                               feat_len=128,
                               num_head=1,
                               fanouts=[10, 15, 20],
                               need_edge_index=0,
                               device="cuda"):
    full_ptr, full_idx = cxgnndl.load_full_graph_structure(dset)
    full_ptr = torch.from_numpy(full_ptr)
    full_idx = torch.from_numpy(full_idx)
    num_all_nodes = full_ptr.shape[0] - 1
    seed_nodes = torch.randint(0, num_all_nodes, (num_seeds, ))
    ptr, idx, input_nodes, num_node_in_layer, num_edge_in_layer = cxgnndl_backend.neighbor_sample(
        full_ptr, full_idx, fanouts, seed_nodes)
    ptr = ptr.to(device)
    idx = idx.to(device)
    if num_head == 1:
        x = torch.randn([input_nodes.shape[0], feat_len],
                        dtype=torch.float32,
                        device=device)
    else:
        x = torch.randn([input_nodes.shape[0], num_head, feat_len],
                        dtype=torch.float32,
                        device=device)
    batch = {}
    batch["num_node_in_layer"] = num_node_in_layer
    batch["num_edge_in_layer"] = num_edge_in_layer
    batch["sub_to_full"] = input_nodes.to(device)
    print(ptr.shape, idx.shape, x.shape, batch["num_node_in_layer"])
    if need_edge_index:
        edge_index = torch.stack([
            idx,
            torch.repeat_interleave(torch.arange(
                0, ptr.shape[0] - 1, device=device),
                                    repeats=ptr[1:] - ptr[:-1])
        ],
                                 dim=0)
        return x, ptr, idx, batch, edge_index
    else:
        return x, ptr, idx, batch
