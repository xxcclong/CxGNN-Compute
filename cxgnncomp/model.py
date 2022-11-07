import torch  # Essential!
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv as  PyG_GCNConv, PyG_SAGEConv, PyG_GINConv, PyG_GATConv
import torch_geometric.nn as pygnn
import dgl.nn.pytorch.conv as dglnn
from .graph_conv import MyGATConv, MyGCNConv, MyRGCNConv, MyRGCNConvNaive, MyRGCNConvOpt1, MyRGCNConvOpt2, MySageConv, MyGINConv
# import torch.autograd.profiler as profiler
# from profile import gpu_profile


class GNN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, graph_type, config, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.graph_type = graph_type
        self.bns = torch.nn.ModuleList()
        self.num_layers = num_layers
        for _ in range(self.num_layers - 1):
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.dropout = dropout
        self.init_convs(**kwargs)

    def init_convs(self, **kwargs):
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            self.init_conv(self.in_channels, self.hidden_channels, **kwargs))
        for _ in range(self.num_layers - 2):
            self.convs.append(
                self.init_conv(self.hidden_channels, self.hidden_channels,
                               **kwargs))
        if self.num_layers > 1:
            self.convs.append(
                self.init_conv(self.hidden_channels, self.out_channels,
                               **kwargs))

    def init_conv(self, in_channels, out_channels, **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward_cxg(self, batch):
        x = batch.x
        for i, conv in enumerate(self.convs[:-1]):
            if self.graph_type == "CSR_Layer":
                num_node = batch.num_node_in_layer[self.num_layers - 1 - i]
            else:
                num_node = 0
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

    def forward(self, input):
        if "CSR" in self.graph_type:
            return self.forward_cxg(input)
        elif "DGL" in self.graph_type:
            return self.forward_dgl(input[0], input[1])
        elif "PyG" in self.graph_type or "COO" in self.graph_type:
            return pygnn.GCNConv(input[0], input[1])
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
            return dglnn.GraphConv(in_channels, out_channels)
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
        if "CSR" in self.graph_type:
            return MyRGCNConv(in_channels,
                              out_channels,
                              num_rel=kwargs["num_rel"])
        elif "DGL" in self.graph_type:
            return dglnn.RelGraphConv(in_channels,
                                      out_channels,
                                      num_rels=kwargs["num_rel"])
        elif "PyG" in self.graph_type or "COO" in self.graph_type:
            return pygnn.RGCNConv(in_channels,
                                  out_channels,
                                  num_relations=kwargs["num_rel"])
        else:
            assert (0)

    def forward_cxg(self, batch):
        x = batch.x
        for i, conv in enumerate(self.convs[:-1]):
            if self.graph_type == "CSR_Layer":
                num_node = batch.num_node_in_layer[self.num_layers - 1 - i]
            else:
                num_node = 0
            etypes = torch.randint(
                0,
                self.num_rel,
                (batch.num_edge_in_layer[self.num_layers - 1 - i], ),
                device=x.device)
            x = conv(x, batch.ptr, batch.idx, etypes, num_node)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        etypes = torch.randint(0,
                               self.num_rel, (batch.num_edge_in_layer[0], ),
                               device=x.device)
        x = self.convs[-1](x, batch.ptr, batch.idx, etypes,
                           batch.num_node_in_layer[0]
                           if self.graph_type == "CSR_Layer" else 0)
        return x.log_softmax(dim=-1)

    def forward_dgl(self, blocks, x):
        for layer, conv in enumerate(self.convs[:-1]):
            etypes = torch.randint(0,
                                   self.num_rel,
                                   (blocks[layer].number_of_edges(), ),
                                   device=x.device)
            x = conv(blocks[layer], x, etypes)
            x = self.bns[layer](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        etypes = torch.randint(0,
                               self.num_rel, (blocks[-1].number_of_edges(), ),
                               device=x.device)
        x = self.convs[-1](blocks[-1], x, etypes)
        return x.log_softmax(dim=-1)


class GAT(GNN):

    def init_conv(self, in_channels, out_channels, **kwargs):
        if 'heads' in kwargs and out_channels % kwargs['heads'] != 0:
            kwargs['heads'] = 1
        if 'concat' not in kwargs or kwargs['concat']:
            out_channels = out_channels // kwargs.get('heads', 1)
        if "CSR" in self.graph_type:
            return MyGATConv(in_channels, out_channels, **kwargs)
        elif "DGL" in self.graph_type:
            return dglnn.GATConv(in_channels,
                                 out_channels,
                                 num_heads=kwargs.get('heads', 1))
        elif "PyG" in self.graph_type or "COO" in self.graph_type:
            return pygnn.GATConv(in_channels, out_channels)
        else:
            assert (0)

    def forward_cxg(self, batch):
        x = batch.x
        assert self.graph_type == "CSR_Layer"
        for i, conv in enumerate(self.convs[:-1]):
            num_dst = batch.num_node_in_layer[self.num_layers - 1 - i]
            num_src = batch.num_node_in_layer[self.num_layers - i]
            num_edge = batch.num_edge_in_layer[self.num_layers - 1 - i]
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


class MLP(GNN):

    def init_conv(self, in_channels, out_channels):
        return torch.nn.Linear(in_channels, out_channels)


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


class RGCN_CSR_Layer(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, graph_type, num_rel, use_etype_schedule):
        super(RGCN_CSR_Layer, self).__init__()
        assert (graph_type in ["CSR_Layer"])
        self.convs = torch.nn.ModuleList()
        self.graph_type = graph_type
        self.use_etype_schedule = use_etype_schedule
        for _ in range(num_layers):
            in_feat = in_channels if _ == 0 else hidden_channels
            out_feat = out_channels if _ == num_layers - 1 else hidden_channels
            if self.use_etype_schedule:
                self.convs.append(MyRGCNConvOpt2(in_feat, out_feat, num_rel))
            else:
                self.convs.append(MyRGCNConvOpt1(in_feat, out_feat, num_rel))
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_rel = num_rel

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, batch):
        x = batch.x
        rel = torch.randint(0,
                            self.num_rel,
                            batch.idx.shape,
                            device=x.device,
                            dtype=torch.int32)
        for i, conv in enumerate(self.convs[:-1]):
            num_node = batch.num_node_in_layer[self.num_layers - 1 - i]
            if i == 0:
                num_used_node = batch.sub_to_full.shape[0]
            else:
                num_used_node = batch.num_node_in_layer[self.num_layers - i]
            if self.use_etype_schedule:
                x = conv(x, batch.etype_partition,
                         batch.typed_num_node_in_layer, num_node, i,
                         self.num_layers)
            else:
                x = conv(x, batch.ptr, batch.idx, rel, num_node, num_used_node)
            # x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.use_etype_schedule:
            x = self.convs[-1](x, batch.etype_partition,
                               batch.typed_num_node_in_layer,
                               batch.num_node_in_layer[0], self.num_layers - 1,
                               self.num_layers)
        else:
            x = self.convs[-1](x, batch.ptr, batch.idx, rel,
                               batch.num_node_in_layer[0],
                               batch.num_node_in_layer[1])
        return x.log_softmax(dim=-1)


graph_type_dict = {
    "cxg": "CSR_Layer",
    "dgl": "DGL",
    "pyg": "PyG",
}

model_dict = {
    "gcn": GCN,
    "gat": GAT,
    "mlp": MLP,
    "gin": GIN,
    "rgcn": RGCN,
    "sage": SAGE,
}


def get_model(config):
    in_channel = config.dl.dataset.feature_dim
    out_channel = config.dl.dataset.num_classes
    hidden_channel = config.train.model.hidden_dim
    num_layers = config.train.model.num_layers
    dropout = config.train.model.dropout
    graph_type = graph_type_dict[config.train.type.lower()]
    if "gat" in config.train.model.type.lower():
        heads = config.train.model.get('heads', 1)
        concat = config.train.model.get('concat', True)
        model = model_dict[config.train.model.type.lower()](in_channel,
                                                            hidden_channel,
                                                            out_channel,
                                                            num_layers,
                                                            dropout,
                                                            graph_type,
                                                            config,
                                                            heads=heads,
                                                            concat=concat)
    elif "rgcn" in config.train.model.type.lower():
        rel = int(config.train.model['num_rel'])
        model = model_dict[config.train.model.type.lower()](in_channel,
                                                            hidden_channel,
                                                            out_channel,
                                                            num_layers,
                                                            dropout,
                                                            graph_type,
                                                            config,
                                                            num_rel=rel)
    else:
        model = model_dict[config.train.model.type.lower()](
            in_channel, hidden_channel, out_channel, num_layers, dropout,
            graph_type, config)
    return model
