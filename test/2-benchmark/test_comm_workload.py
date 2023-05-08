import torch
import cxgnncomp as cxgc
import time
import cxgnncomp_backend
import argparse


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
    cache=None,
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
            if cache is not None:
                sub_edge_index = edge_index[:, (
                    (cache[edge_index[0]]) &
                    (torch.div(edge_index[0],
                               num_node_per_device,
                               rounding_mode='floor') == i)
                ) & ((torch.div(
                    edge_index[1], num_node_per_device, rounding_mode='floor'
                ) == j))]
            else:
                sub_edge_index = edge_index[:, (torch.div(
                    edge_index[0], num_node_per_device, rounding_mode='floor'
                ) == i) & ((torch.div(
                    edge_index[1], num_node_per_device, rounding_mode='floor'
                ) == j))]

            # if cache is not None:
            #     sub_edge_index2 = edge_index[:, (
            #         (cache[edge_index[1]]) &
            #         (torch.div(edge_index[0],
            #                    num_node_per_device,
            #                    rounding_mode='floor') == j)
            #     ) & ((torch.div(
            #         edge_index[1], num_node_per_device, rounding_mode='floor'
            #     ) == i))]
            # else:
            #     sub_edge_index2 = edge_index[:, (torch.div(
            #         edge_index[0], num_node_per_device, rounding_mode='floor'
            #     ) == j) & ((torch.div(
            #         edge_index[1], num_node_per_device, rounding_mode='floor'
            #     ) == i))]
            # sub_edge_index = torch.cat([sub_edge_index, sub_edge_index2],
            #                            dim=1)
            # del sub_edge_index2

            # sub_edge_index = sub_edge_index.cuda()
            differentiaed = False
            if differentiaed:
                degree_bias_num = analyze_sub_edge_index(
                    edge_index=sub_edge_index)
            else:
                degree_bias_num = None
            arr.append([
                int(torch.unique(sub_edge_index[0]).shape[0]) /
                num_node_per_device,
                int(torch.unique(sub_edge_index[1]).shape[0]) /
                num_node_per_device
            ])
            del sub_edge_index
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
    comms1 = [0 for i in range(num_device)]
    comms2 = [0 for i in range(num_device)]
    for i in range(len(comm_mat)):
        for j in range(len(comm_mat[i])):
            if i == j:
                continue
            comm1 += comm_mat[i][j][0]
            comm2 += min(comm_mat[i][j][:2])
            comm3 += max(comm_mat[i][j][:2])
            comms1[i] += comm_mat[i][j][0]
            comms2[i] += min(comm_mat[i][j][:2])
    print(comm1, comm2, comm3)
    print(comms1, comms2)


torch.set_printoptions(precision=3)

parser = argparse.ArgumentParser()
parser.add_argument("--num_device", type=int, default=4)
parser.add_argument("--undirected", type=bool, default=False)
parser.add_argument("--dset", type=str)
parser.add_argument("--cache", type=float, default=0)
parser.add_argument("--reorder", action="store_true")
args = parser.parse_args()
print(args)

dset = args.dset

if dset not in ["papers100M", "mag240m", "wiki90m"]:
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
    edge_index = edge_index.cuda().to(torch.int32)
    print("self loop", (edge_index[0] == edge_index[1]
                        ).any())  # examine whether self-loop exists
    num_node = int(torch.max(edge_index) + 1)
    num_edge = edge_index.shape[1]
    print(torch.max(edge_index), torch.min(edge_index))
    if args.undirected:
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    if args.cache > 0:
        for_deg = edge_index[0][torch.randperm(num_edge)]
        torch.cuda.synchronize()
        print("begin deg")
        deg = torch.bincount(for_deg)
        del for_deg
        torch.cuda.synchronize()
        print("end deg")
    num_edge = edge_index.shape[1]
    idx = edge_index[1]

num_device = 4
print(f"num_edge {num_edge} num_node {num_node} num_device {num_device}")

# deg = torch.randn([deg.shape[0]], device=deg.device)
percentage = args.cache
if percentage == 0:
    cache_status = None
else:
    sorted, indices = torch.sort(deg, descending=True)
    thres = deg[indices[int(num_node * percentage)]]
    print("thres", thres)
    cache_status = deg < thres
    print("num uncached", torch.sum(cache_status))
    del deg

print(
    torch.cuda.get_device_properties(0).total_memory,
    torch.cuda.memory_reserved(0), torch.cuda.memory_allocated(0))

print("==========Origin==========")
get_count(edge_index, num_node, num_device, cache=cache_status)

print(
    torch.cuda.get_device_properties(0).total_memory,
    torch.cuda.memory_reserved(0), torch.cuda.memory_allocated(0))

if args.reorder:
    print("==========Rabbit==========")
    reorder_metric = torch.from_numpy(
        np.fromfile(
            f"/home/huangkz/repos/rabbit_order/demo/reorder_{args.dset}.dat",
            dtype=np.int64)).to(torch.int32).cuda()
    edge_index = reorder_metric[edge_index]
    del reorder_metric
    get_count(edge_index, num_node, num_device, cache=cache_status)

    print(
        torch.cuda.get_device_properties(0).total_memory,
        torch.cuda.memory_reserved(0), torch.cuda.memory_allocated(0))

print("==========Random==========")
reorder_metric = torch.randperm(num_node,
                                device=edge_index.device,
                                dtype=torch.int32)
edge_index = reorder_metric[edge_index]
del reorder_metric

get_count(edge_index, num_node, num_device, cache=cache_status)
