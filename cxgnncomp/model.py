import torch  # Essential!
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
from .graph_conv import MyGATConv, MyGCNConv, MyRGCNConvNaive, MyRGCNConvOpt1, MyRGCNConvOpt2, MySageConv, MyGINConv
# import torch.autograd.profiler as profiler
# from profile import gpu_profile


class GNN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, graph_type, config, **kwargs):
        super().__init__()

        self.graph_type = graph_type
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            self.init_conv(in_channels, hidden_channels, **kwargs))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        if num_layers > 1:
            self.convs.append(
                self.init_conv(hidden_channels, out_channels, **kwargs))
        self.num_layers = num_layers
        self.dropout = dropout

    def init_conv(self, in_channels, out_channels, **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, batch):
        x = batch.x
        for i, conv in enumerate(self.convs[:-1]):
            if "CSR" in self.graph_type:
                if self.graph_type == "CSR_Layer":
                    num_node = batch.num_node_in_layer[self.num_layers - 1 - i]
                else:
                    num_node = 0
                x = conv(x, batch.ptr, batch.idx, num_node)
            else:
                x = conv(x, batch.edge_index)
            # TODO: BN has different results for layered implementation
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if "CSR" in self.graph_type:
            x = self.convs[-1](x, batch.ptr, batch.idx,
                               batch.num_node_in_layer[0]
                               if self.graph_type == "CSR_Layer" else 0)
        else:
            x = self.convs[-1](x, batch.edge_index)
        return x.log_softmax(dim=-1)


class SAGE(GNN):
    noise = False

    def init_conv(self, in_channels, out_channels, **kwargs):
        if "CSR" in self.graph_type:
            return MySageConv(in_channels, out_channels)
        else:
            return SAGEConv(in_channels, out_channels)

    def set_trans_optim(self):
        self.convs[0].mean_forward = False

    def forward(self, batch):
        x = batch.x
        for i, conv in enumerate(self.convs[:-1]):
            if "CSR" in self.graph_type:
                if self.graph_type == "CSR_Layer":
                    num_node = batch.num_node_in_layer[self.num_layers - 1 - i]
                else:
                    num_node = 0
                x = conv(x,
                         batch.ptr,
                         batch.idx,
                         num_node,
                         noise=self.noise and self.training)
            else:
                x = conv(x, batch.edge_index)
            # TODO: BN has different results for layered implementation
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if "CSR" in self.graph_type:
            x = self.convs[-1](x, batch.ptr, batch.idx,
                               batch.num_node_in_layer[0]
                               if self.graph_type == "CSR_Layer" else 0)
        else:
            x = self.convs[-1](x, batch.edge_index)
        return x.log_softmax(dim=-1)


class GCN(GNN):

    def init_conv(self, in_channels, out_channels, **kwargs):
        if "CSR" in self.graph_type:
            return MyGCNConv(in_channels, out_channels)
        else:
            return GCNConv(in_channels, out_channels)


class GAT(GNN):

    def init_conv(self, in_channels, out_channels, **kwargs):
        if 'heads' in kwargs and out_channels % kwargs['heads'] != 0:
            kwargs['heads'] = 1
        if 'concat' not in kwargs or kwargs['concat']:
            out_channels = out_channels // kwargs.get('heads', 1)
        if "CSR" in self.graph_type:
            return MyGATConv(in_channels, out_channels, **kwargs)
        else:
            return GATConv(in_channels, out_channels, **kwargs)

    def forward(self, batch):
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


class MLP(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, graph_type, config):
        super(MLP, self).__init__()
        self.graph_type = graph_type
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, batch):
        x = batch.x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)


class GIN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, graph_type, config):
        super(GIN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.graph_type = graph_type
        if "CSR" in self.graph_type:
            self.convs.append(MyGINConv(in_channels, hidden_channels))
        else:
            self.convs.append(GINConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            if "CSR" in self.graph_type:
                self.convs.append(MyGINConv(hidden_channels, hidden_channels))
            else:
                self.convs.append(GINConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        if "CSR" in self.graph_type:
            self.convs.append(MyGINConv(hidden_channels, out_channels))
        else:
            self.convs.append(GINConv(hidden_channels, out_channels))
        self.dropout = dropout
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, batch):
        x = batch.x
        for i, conv in enumerate(self.convs[:-1]):
            if "CSR" in self.graph_type:
                x = conv(
                    x, batch.ptr, batch.idx,
                    batch.num_node_in_layer[self.num_layers - 1 - i]
                    if self.graph_type == "CSR_Layer" else 0)
            else:
                x = conv(x, batch.edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if "CSR" in self.graph_type:
            x = self.convs[-1](x, batch.ptr, batch.idx,
                               batch.num_node_in_layer[0]
                               if self.graph_type == "CSR_Layer" else 0)
        else:
            x = self.convs[-1](x, batch.edge_index)
        return x.log_softmax(dim=-1)


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
