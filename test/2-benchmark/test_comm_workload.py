import torch
import cxgnncomp as cxgc
import time
import cxgnncomp_backend


def analyze_sub_edge_index(edge_index):
    edge_index[1] = edge_index[1] - torch.min(edge_index[1])
    edge_index[0] = edge_index[0] - torch.min(edge_index[0])
    num_dst = torch.unique(edge_index[1]).shape[0]
    num_src = torch.unique(edge_index[0]).shape[0]
    # print(num_dst, num_src)
    deg_dst = torch.bincount(edge_index[1])
    deg_src = torch.bincount(edge_index[0])
    visited_dst = torch.zeros(int(torch.max(edge_index[1])) + 1,
                              dtype=torch.bool)
    visited_src = torch.zeros(int(torch.max(edge_index[0])) + 1,
                              dtype=torch.bool)
    num = 0
    for i in range(edge_index.shape[1]):
        dst = edge_index[1, i]
        src = edge_index[0, i]
        if visited_dst[dst] or visited_src[src]:
            continue
        if deg_dst[dst] > deg_src[src]:
            visited_dst[dst] = True
            num += 1
        else:
            visited_src[src] = True
            num += 1
    # print(num)
    return num


def get_count(
    edge_index,
    num_node,
    num_device,
):
    num_node_per_device = num_node // num_device
    count = torch.zeros([num_device, num_device], dtype=torch.int64)
    for i in range(num_device):
        for j in range(num_device):
            count[i, j] = torch.sum((torch.div(
                edge_index[0], num_node_per_device, rounding_mode='floor') == i
                                     )
                                    & (torch.div(edge_index[1],
                                                 num_node_per_device,
                                                 rounding_mode='floor') == j))

            # count[i, j] = torch.sum((edge_index[0] % num_device == i)
            #                         & (edge_index[1] % num_device == j))
    print(count / num_edge)
    print(num_edge)
    print(torch.sum(torch.sum(count, dim=1), dim=0))

    # edge_index = edge_index.cuda()
    comm_mat = []
    for i in range(num_device):
        arr = []
        for j in range(num_device):
            sub_edge_index = edge_index[:, (torch.div(
                edge_index[0], num_node_per_device, rounding_mode='floor'
            ) == i) & (
                (torch.
                 div(edge_index[1], num_node_per_device, rounding_mode='floor'
                     ) == j))]
            sub_edge_index = sub_edge_index.cuda()
            differentiaed = False
            if differentiaed:
                degree_bias_num = analyze_sub_edge_index(
                    edge_index=sub_edge_index)
            else:
                degree_bias_num = None
            arr.append([
                torch.unique(sub_edge_index[0]).shape[0] / num_node_per_device,
                torch.unique(sub_edge_index[1]).shape[0] / num_node_per_device
            ])
            if degree_bias_num is not None:
                arr[-1].append(degree_bias_num / num_node_per_device)
            else:
                arr[-1].append(10000)
            if j != num_device - 1:
                print(
                    f"{round(arr[-1][0], 3)},{round(arr[-1][1], 3)},{round(arr[-1][2], 3)}",
                    end="\t")
            else:
                print(
                    f"{round(arr[-1][0], 3)},{round(arr[-1][1], 3)},{round(arr[-1][2], 3)}"
                )
        comm_mat.append(arr)

    comm1 = 0
    comm2 = 0
    comm3 = 0
    for i in range(len(comm_mat)):
        for j in range(len(comm_mat[i])):
            if i == j:
                continue
            comm1 += comm_mat[i][j][0]
            comm2 += min(comm_mat[i][j][:2])
            comm3 += min(comm_mat[i][j])
    print(comm1, comm2, comm3)


torch.set_printoptions(precision=3)

# dset = "arxiv"
dset = "papers100M"
# dset = "mag240m"

if dset not in ["papers100M", "mag240m"]:
    x, ptr, idx, batch, edge_index = cxgc.prepare_data_full_graph(
        dset=dset, need_edge_index=True, need_feat=False, undirected=True)
    num_edge = idx.shape[0]
    num_node = ptr.shape[0] - 1
    deg = ptr[1:] - ptr[:-1]
else:
    import numpy as np
    edge_index = np.fromfile(
        f"/home/huangkz/data/dataset_diskgnn/{dset}/processed/edge_index.dat",
        dtype=np.int64)
    edge_index = torch.from_numpy(edge_index).view(2, -1)
    print((edge_index[0] == edge_index[1]).any())
    num_node = int(torch.max(edge_index) + 1)
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]],
                           dim=1)  # undirected
    deg = torch.bincount(edge_index[0], minlength=num_node)
    num_edge = edge_index.shape[1]
    ptr = torch.bincount(input=edge_index[0], minlength=num_node)
    idx = edge_index[1]

num_device = 8
print(f"num_edge {num_edge} num_node {num_node} num_device {num_device}")

get_count(edge_index, num_node, num_device)

assert deg.shape[0] == num_node
sorted, indices = torch.sort(deg, descending=True)
new_indices = torch.zeros_like(indices)
new_indices[indices] = torch.arange(num_node, device=indices.device)
edge_index = new_indices[edge_index]
# print(sorted)
# print(torch.bincount(edge_index[0]))

get_count(edge_index, num_node, num_device)