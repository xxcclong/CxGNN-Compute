import numpy as np
import torch
import os
import dgl


class Batch():

    def __init__(self,
                 x,
                 ptr,
                 idx,
                 num_node_in_layer=None,
                 num_edge_in_layer=None,
                 target=None):
        self.x = x
        self.ptr = ptr
        self.idx = idx
        self.num_node_in_layer = num_node_in_layer
        self.num_edge_in_layer = num_edge_in_layer
        self.target = target
        self.block = None

    def to(self, dev):
        if self.x is not None:
            self.x = self.x.to(dev)
        if self.ptr is not None:
            self.ptr = self.ptr.to(dev)
        if self.idx is not None:
            self.idx = self.idx.to(dev)
        if self.target is not None:
            self.target = self.target.to(dev)
        if self.block is not None:
            self.block = self.block.to(dev)

    def tofile(self, dir):
        if self.x is not None:
            self.x.cpu().numpy().tofile(dir + "/x.dat")
        if self.ptr is not None:
            self.ptr.cpu().numpy().tofile(dir + "/ptr.dat")
        if self.idx is not None:
            self.idx.cpu().numpy().tofile(dir + "/idx.dat")
        if self.target is not None:
            self.target.cpu().numpy().tofile(dir + "/target.dat")

    def fromfile(self, dir):
        if os.path.exists(dir + "/x.dat"):
            self.x = torch.from_numpy(
                np.fromfile(dir + "/x.dat", dtype=np.float32))
        if os.path.exists(dir + "/ptr.dat"):
            self.ptr = torch.from_numpy(
                np.fromfile(dir + "/ptr.dat", dtype=np.int64))
        else:
            print(dir + "/ptr.dat", "not found")
        if os.path.exists(dir + "/idx.dat"):
            self.idx = torch.from_numpy(
                np.fromfile(dir + "/idx.dat", dtype=np.int64))
        if os.path.exists(dir + "/target.dat"):
            self.target = torch.from_numpy(
                np.fromfile(dir + "/target.dat", dtype=np.int64))

    def todgl(self, num_src, num_dst):
        new_ptr = torch.ones(
            num_dst + 1, dtype=self.ptr.dtype, device=self.ptr.device) * -1
        new_ptr[self.target + 1] = self.ptr[1:]
        new_ptr[0] = 0
        new_ptr = torch.from_numpy(np.maximum.accumulate(
            new_ptr.cpu().numpy())).to(self.ptr.device)
        # for i in range(1, num_dst + 1):
        #     if new_ptr[i] == -1:
        #         new_ptr[i] = new_ptr[i - 1]

        self.block = dgl.create_block(
            ('csc',
             (new_ptr, self.idx, torch.tensor([], dtype=self.ptr.dtype))),
            num_src, num_dst)

    def totype(self, dtype):
        self.ptr = self.ptr.to(dtype)
        self.idx = self.idx.to(dtype)
        self.target = self.target.to(dtype)


class PyGBatch():

    def __init__(self, x, edge_index):
        self.x = x
        self.edge_index = edge_index
