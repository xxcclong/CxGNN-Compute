import torch
import numpy as np

dset = "arxiv"


def to_graph(dset):
    edge_index = np.fromfile(
        f"/home/huangkz/data/dataset_diskgnn/{dset}/processed/edge_index.dat",
        dtype=np.int64).reshape(2, -1)
    num_edge = edge_index.shape[1]
    f = open(
        f"/home/huangkz/data/dataset_diskgnn/{dset}/processed/edge_index.txt",
        "w")
    s = ""
    for i in range(num_edge):
        s += f"{edge_index[0, i]} {edge_index[1, i]}\n{edge_index[1, i]} {edge_index[0, i]}\n"
    f.write(s)
    f.close()


def to_mesh(dset):
    ptr = np.fromfile(
        f"/home/huangkz/data/dataset_diskgnn/{dset}/processed/csr_ptr_undirected.dat",
        dtype=np.int64)

    idx = np.fromfile(
        f"/home/huangkz/data/dataset_diskgnn/{dset}/processed/csr_idx_undirected.dat",
        dtype=np.int64)
    num_node = ptr.shape[0] - 1
    num_edge = idx.shape[0]
    f = open(
        f"/home/huangkz/data/dataset_diskgnn/{dset}/processed/edge_index.mesh",
        "w")
    f.write(f"{num_node} {num_edge}\n")
    print(f"{num_node} {num_edge}")
    for i in range(num_node):
        start = ptr[i]
        end = ptr[i + 1]
        s = ""
        for j in range(start, end):
            s += str(idx[j] + 1) + ' '
        f.write(s + "\n")
    f.close()


# to_mesh(dset)


def prof_sort(dset):
    ptr = np.fromfile(
        f"/home/huangkz/data/dataset_diskgnn/{dset}/processed/csr_ptr_undirected.dat",
        dtype=np.int64)
    idx = np.fromfile(
        f"/home/huangkz/data/dataset_diskgnn/{dset}/processed/csr_idx_undirected.dat",
        dtype=np.int64)
    import time
    start = time.time()
    sorted, indices = torch.sort(torch.from_numpy(ptr))
    print(time.time() - start)
    num_node = ptr.shape[0] - 1
    num_edge = idx.shape[0]

    # import cxgnncomp_backend
    # rel = torch.randint(0, 7, (num_edge, ), dtype=torch.int32)
    ptr = torch.from_numpy(ptr).cuda()
    deg = ptr[1:] - ptr[:-1]
    output = torch.cumsum(deg, dim=0)
    assert torch.all(output == ptr[1:])
    torch.cumsum(deg, dim=0)
    torch.cumsum(deg, dim=0)
    torch.cuda.synchronize()
    import cxgnncomp

    start = time.time()
    # cxgnncomp_backend.rel_schedule(ptr, idx, rel)
    # cxgnncomp.neighbor_grouping(ptr)
    # torch.cumsum(deg, dim=0)
    cxgnncomp.neighbor_grouping_gpu(ptr)
    torch.cuda.synchronize()
    print(time.time() - start)

    start = time.time()
    sorted, indices = torch.sort(ptr)
    torch.cuda.synchronize()
    print(time.time() - start)


    start = time.time()
    # arr = []
    # for i in range(num_node):
    #     pos = indices[i]
    #     start = ptr[pos]
    #     end = ptr[pos + 1]
    #     arr.extend(sorted[idx[start:end]])
    print(time.time() - start)


prof_sort("papers100M")