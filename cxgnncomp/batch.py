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

    def to(self, dev):
        if self.x is not None:
            self.x = self.x.to(dev)
        if self.ptr is not None:
            self.ptr = self.ptr.to(dev)
        if self.idx is not None:
            self.idx = self.idx.to(dev)
        if self.target is not None:
            self.target = self.target.to(dev)



class PyGBatch():

    def __init__(self, x, edge_index):
        self.x = x
        self.edge_index = edge_index
