import torch  # Essential!
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv as  PyG_GCNConv, PyG_SAGEConv, PyG_GINConv, PyG_GATConv
import torch_geometric.nn as pygnn
import dgl.nn.pytorch.conv as dglnn
from .graph_conv import MyGATConv, MyGCNConv, MyRGCNConv, MySageConv, MyGINConv
from .util import log
# import torch.autograd.profiler as profiler
# from profile import gpu_profile
import cxgnncomp_backend
import dgl
import dgl.function as fn
from .neighbor_lstm import NeighborLstmPadOP, NeighborLstmOP


class GNN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layer,
                 dropout, graph_type, config, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.graph_type = graph_type
        self.bns = torch.nn.ModuleList()
        self.num_layer = num_layer
        for _ in range(self.num_layer - 1):
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.dropout = dropout
        self.init_convs(**kwargs)

    def init_convs(self, **kwargs):
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            self.init_conv(self.in_channels, self.hidden_channels, layer=0, **kwargs))
        for i in range(self.num_layer - 2):
            self.convs.append(
                self.init_conv(self.hidden_channels, self.hidden_channels, layer=i + 1,
                               **kwargs))
        if self.num_layer > 1:
            self.convs.append(
                self.init_conv(self.hidden_channels, self.out_channels, layer=self.num_layer - 1,
                               **kwargs))

    def init_conv(self, in_channels, out_channels, **kwargs):
        pass
        # raise NotImplementedError

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward_cxg(self, batch, skip_first=False):
        x = batch.x
        # print(f"skip_first {skip_first} {x.shape} {batch.num_node_in_layer}")
        for i, conv in enumerate(self.convs[:-1]):
            if self.graph_type == "CSR_Layer":
                num_node = batch.num_node_in_layer[self.num_layer - 1 - i]
            else:
                num_node = 0
            # print("num_node in layer", batch.num_node_in_layer, batch.x.shape)
            if (not skip_first) or i != 0:
                x = conv(x, batch.ptr, batch.idx, num_node)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, batch.ptr, batch.idx, batch.num_node_in_layer[0]
                           if self.graph_type == "CSR_Layer" else 0)
        return x.log_softmax(dim=-1)

    def forward_dgl(self, blocks, x):
        for layer, conv in enumerate(self.convs[:-1]):
            x = conv(blocks[layer], x)
            x = self.bns[layer](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](blocks[-1], x)
        return x.log_softmax(dim=-1)

    def forward_pyg(self, edge_index, x):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)

    def forward(self, input, skip_first=False):
        if "CSR" in self.graph_type:
            return self.forward_cxg(input)
        elif "DGL" in self.graph_type:
            return self.forward_dgl(input[0], input[1])
        elif "PyG" in self.graph_type or "COO" in self.graph_type:
            return self.forward_pyg(input.edge_index, input.x)
        else:
            assert (0)


class SAGE(GNN):

    def init_conv(self, in_channels, out_channels, **kwargs):
        if "CSR" in self.graph_type:
            return MySageConv(in_channels, out_channels)
        elif "DGL" in self.graph_type:
            return dglnn.SAGEConv(in_channels,
                                  out_channels,
                                  aggregator_type="mean")
        elif "PyG" in self.graph_type or "COO" in self.graph_type:
            return pygnn.SAGEConv(in_channels, out_channels)
        else:
            assert (0)


class GCN(GNN):

    def init_conv(self, in_channels, out_channels, **kwargs):
        if "CSR" in self.graph_type:
            return MyGCNConv(in_channels, out_channels)
        elif "DGL" in self.graph_type:
            return dglnn.GraphConv(in_channels,
                                   out_channels,
                                   allow_zero_in_degree=True)
        elif "PyG" in self.graph_type or "COO" in self.graph_type:
            return pygnn.GCNConv(in_channels, out_channels)
        else:
            assert (0)


# class GCN2(GNN):
#     def init_conv(self, in_channels, out_channels, **kwargs):
#         if "CSR" in self.graph_type:
#             raise NotImplementedError
#         elif "DGL" in self.graph_type:
#             return dglnn.GCN2Conv(in_channels, out_channels)
#         elif "PyG" in self.graph_type or "COO" in self.graph_type:
#             raise NotImplementedError
#         else:
#             assert (0)


class RGCN(GNN):

    def init_conv(self, in_channels, out_channels, **kwargs):
        self.num_rel = kwargs["num_rel"]
        self.dataset_name = kwargs["dataset_name"]
        self.gen_rel = self.dataset_name == "rmag240m"
        if self.gen_rel:
            self.num_rel = 5
            log.warn("Generate relation type for RMAG240M at 5")
        if "CSR" in self.graph_type:
            return MyRGCNConv(in_channels, out_channels, num_rel=self.num_rel)
        elif "DGL" in self.graph_type:
            return dglnn.RelGraphConv(in_channels,
                                      out_channels,
                                      num_rels=self.num_rel)
        elif "PyG" in self.graph_type or "COO" in self.graph_type:
            return pygnn.RGCNConv(in_channels,
                                  out_channels,
                                  num_relations=self.num_rel)
        else:
            assert (0)

    def forward_cxg(self, batch, skip_first):
        x = batch.x
        if self.gen_rel:
            etypes = cxgnncomp_backend.gen_edge_type_mag240m(
                batch.ptr, batch.idx, batch.sub_to_full)
        else:
            etypes = torch.randint(
                0,
                self.num_rel, (batch.num_edge_in_layer[self.num_layer - 1], ),
                device=x.device)
        for i, conv in enumerate(self.convs[:-1]):
            if self.graph_type == "CSR_Layer":
                num_node = batch.num_node_in_layer[self.num_layer - 1 - i]
            else:
                num_node = 0
            x = conv(x, batch.ptr, batch.idx,
                     etypes[:batch.num_edge_in_layer[self.num_layer - 1 - i]],
                     num_node)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, batch.ptr, batch.idx,
                           etypes[:batch.num_edge_in_layer[0]],
                           batch.num_node_in_layer[0]
                           if self.graph_type == "CSR_Layer" else 0)
        return x.log_softmax(dim=-1)

    def forward_dgl(self, blocks, x):
        etypes = torch.randint(0,
                               self.num_rel, (blocks[0].number_of_edges(), ),
                               device=x.device)
        for layer, conv in enumerate(self.convs[:-1]):
            x = conv(blocks[layer], x,
                     etypes[:blocks[layer].number_of_edges()])
            x = self.bns[layer](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](blocks[-1], x,
                           etypes[:blocks[-1].number_of_edges()])
        return x.log_softmax(dim=-1)

    def forward_pyg(self, edge_index, x):
        etypes = torch.randint(0,
                               self.num_rel, (edge_index.shape[1], ),
                               device=x.device)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, etypes)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, etypes)
        return x.log_softmax(dim=-1)


class GAT(GNN):

    def init_conv(self, in_channels, out_channels, **kwargs):
        if 'heads' in kwargs and out_channels % kwargs['heads'] != 0:
            kwargs['heads'] = 1
        if 'concat' in kwargs and kwargs['concat']:
            out_channels = out_channels // kwargs.get('heads', 1)
        if "CSR" in self.graph_type:
            return MyGATConv(in_channels, out_channels, **kwargs)
        elif "DGL" in self.graph_type:
            return dglnn.GATConv(in_channels,
                                 out_channels,
                                 allow_zero_in_degree=True,
                                 num_heads=kwargs.get('heads', 1))
        elif "PyG" in self.graph_type or "COO" in self.graph_type:
            return pygnn.GATConv(in_channels, out_channels)
        else:
            assert (0)

    def forward_dgl(self, blocks, x):
        for layer, conv in enumerate(self.convs[:-1]):
            x = conv(blocks[layer], x)
            x = x.mean(dim=1)
            x = self.bns[layer](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](blocks[-1], x)
        x = x.mean(dim=1)
        return x.log_softmax(dim=-1)

    def forward_cxg(self, batch):
        x = batch.x
        assert self.graph_type == "CSR_Layer"
        for i, conv in enumerate(self.convs[:-1]):
            num_dst = batch.num_node_in_layer[self.num_layer - 1 - i]
            num_src = batch.num_node_in_layer[self.num_layer - i]
            num_edge = batch.num_edge_in_layer[self.num_layer - 1 - i]
            x = conv(x,
                     batch.ptr,
                     batch.idx,
                     num_dst=num_dst,
                     num_src=num_src,
                     num_edge=num_edge)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x,
                           batch.ptr,
                           batch.idx,
                           num_dst=batch.num_node_in_layer[0],
                           num_src=batch.num_node_in_layer[1],
                           num_edge=batch.num_edge_in_layer[0])
        return x.log_softmax(dim=-1)


class GIN(GNN):

    def init_conv(self, in_channels, out_channels, **kwargs):
        if "CSR" in self.graph_type:
            return MyGINConv(in_channels, out_channels)
        elif "DGL" in self.graph_type:
            return dglnn.GINConv(torch.nn.Linear(in_channels, out_channels),
                                 aggregator_type='mean')
        elif "PyG" in self.graph_type or "COO" in self.graph_type:
            return pygnn.GINConv(torch.nn.Linear(in_channels, out_channels))
        else:
            assert (0)

class LSTMConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, graph_type) -> None:
        super().__init__()
        self.graph_type = graph_type
        if graph_type == "PyG":
            self.conv = pygnn.SAGEConv(in_channels, out_channels, aggr="lstm").cuda()
        else:
            self.lstm_module = torch.nn.LSTM(in_channels, out_channels, batch_first=True).cuda()
        self.count = None
        self.previous_num_node = -1

    def lstm_dgl(self, batch):
        def _lstm_reducer(nodes):
            m = nodes.mailbox["m"]  # (B, L, D)
            batch_size = m.shape[0]
            _, (rst, _) = self.lstm_module(
                m,
            )
            return {"neigh": rst.squeeze(0)}

        num_src = batch.num_node_in_layer[-1]
        num_dst = batch.num_node_in_layer[-2]
        dgl_graph = dgl.graph((batch.edge_index[0], batch.edge_index[1]), num_nodes=num_src).to("cuda")
        # dgl_graph.ndata["h"] = batch.x
        dgl_graph.srcdata["h"] = batch.x
        msg_fn = fn.copy_u("h", "m")
        # msg_fn = fn.copy_src("h", "m")
        dgl_graph.update_all(msg_fn, _lstm_reducer)
        return dgl_graph.ndata["neigh"]

    def forward(self, batch):
        with torch.inference_mode():
            if "CSR" in self.graph_type:
                if self.count is None or self.previous_num_node != batch.ptr.shape[0] - 1:
                    deg = batch.ptr[1:] - batch.ptr[:-1]
                    self.count = torch.bincount(deg).cpu()
                    self.previous_num_node = batch.ptr.shape[0] - 1
                num_edge = batch.idx.shape[0]
                if num_edge / 2 > 1e8 and num_edge / 2 < 1.2e8:
                    num_center_in_batch = 4096 * 4
                    num_neighbor_in_batch = 50000 * 4
                else:
                    num_center_in_batch = 4096
                    num_neighbor_in_batch = 50000
                # num_center_in_batch = 64
                # num_neighbor_in_batch = 10000
                # print(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
                return NeighborLstmPadOP(
                    self.lstm_module, batch.ptr, batch.idx, batch.x, self.count, num_center_in_batch, num_neighbor_in_batch)
                # return NeighborLstmOP(
                #     self.lstm_module, batch.ptr, batch.idx, batch.x, self.count)
            elif "DGL" in self.graph_type:
                return self.lstm_dgl(batch)
            elif "PyG" in self.graph_type or "COO" in self.graph_type:
                return self.conv(batch.edge_index)


graph_type_dict = {
    "cxg": "CSR_Layer",
    "dgl": "DGL",
    "pyg": "PyG",
}

model_dict = {
    "gcn": GCN,
    "gat": GAT,
    "gin": GIN,
    "rgcn": RGCN,
    "sage": SAGE,
}


def get_model(config):
    in_channel = config.dl.dataset.feature_dim
    out_channel = config.dl.dataset.num_classes
    hidden_channel = config.train.model.hidden_dim
    num_layer = config.dl.sampler.train.num_layer
    dropout = config.train.model.dropout
    graph_type = graph_type_dict[config.train.type.lower()]
    if "gat" in config.train.model.type.lower():
        heads = config.train.model.get('heads', 1)
        concat = config.train.model.get('concat', False)
        model = model_dict[config.train.model.type.lower()](in_channel,
                                                            hidden_channel,
                                                            out_channel,
                                                            num_layer,
                                                            dropout,
                                                            graph_type,
                                                            config,
                                                            heads=heads,
                                                            concat=concat)
    elif "rgcn" in config.train.model.type.lower():
        rel = int(config.train.model['num_rel'])
        model = model_dict[config.train.model.type.lower()](
            in_channel,
            hidden_channel,
            out_channel,
            num_layer,
            dropout,
            graph_type,
            config,
            num_rel=rel,
            dataset_name=config.dl.dataset.name.lower())
    else:
        model = model_dict[config.train.model.type.lower()](in_channel,
                                                            hidden_channel,
                                                            out_channel,
                                                            num_layer, dropout,
                                                            graph_type, config)
    return model


def get_conv_from_str(model_str, infeat, outfeat, num_head=-1, num_rel=-1):
    model_str = model_str.lower()
    if model_str == "gat":
        conv = MyGATConv(infeat, outfeat, heads=num_head)
    elif model_str == "gcn":
        conv = MyGCNConv(infeat, outfeat)
    elif model_str == "sage":
        conv = MySageConv(infeat, outfeat)
    elif model_str == "rgcn":
        conv = MyRGCNConv(infeat, outfeat, num_rel=num_rel)
    else:
        assert False, f"unknown model {model_str}"
    return conv


def get_model_from_str(mtype,
                       infeat,
                       hiddenfeat,
                       outfeat,
                       graph_type,
                       num_layer,
                       num_head=-1,
                       num_rel=-1,
                       dataset=None,
                       dropout=0.5):
    mtype = mtype.upper()
    if mtype == "GCN":
        model = GCN(infeat,
                    hiddenfeat,
                    outfeat,
                    num_layer,
                    dropout=dropout,
                    graph_type=graph_type,
                    config=None)
    elif mtype == "GAT":
        model = GAT(infeat,
                    hiddenfeat,
                    outfeat,
                    num_layer,
                    dropout=dropout,
                    graph_type=graph_type,
                    config=None,
                    heads=num_head)
    elif mtype == "SAGE":
        model = SAGE(infeat,
                     hiddenfeat,
                     outfeat,
                     num_layer,
                     dropout=dropout,
                     graph_type=graph_type,
                     config=None)
    elif mtype == "RGCN":
        model = RGCN(infeat,
                     hiddenfeat,
                     outfeat,
                     num_layer,
                     dropout=dropout,
                     graph_type=graph_type,
                     config=None,
                     num_rel=num_rel,
                     dataset_name=dataset)
    elif mtype == "LSTM":
        model = LSTMConv(infeat, outfeat, graph_type)
    else:
        assert False, f"unknown model {mtype}"
    return model
