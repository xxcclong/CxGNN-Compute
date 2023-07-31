import torch
import numpy as np

dset = "friendster"

edge_index = np.fromfile(
    f"/home/huangkz/data/dataset_diskgnn/{dset}/processed/edge_index.dat",
    dtype=np.int64).reshape(2, -1)
num_edge = edge_index.shape[1]
f = open(f"/home/huangkz/data/dataset_diskgnn/{dset}/processed/edge_index.txt",
         "w")
s = ""
for i in range(num_edge):
    s += f"{edge_index[0, i]} {edge_index[1, i]}\n{edge_index[1, i]} {edge_index[0, i]}\n"
f.write(s)
f.close()
