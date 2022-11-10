import torch
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot
from cxgnncomp_backend import edge_attention, sage_sum_forward_edge_value, gather, sage_sum_forward, aggr_rel, sage_mean_forward, selective_aggr, selective_aggr_bwd, aggr_rgcn_direct_func

torch.fx.wrap("edge_attention")
torch.fx.wrap("sage_sum_forward_edge_value")
torch.fx.wrap("gather")
torch.fx.wrap("sage_sum_forward")
torch.fx.wrap("aggr_rel")
torch.fx.wrap("sage_mean_forward")


class MySageConv(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 root_weight: bool = True,
                 bias: bool = True,
                 mean_forward: bool = True):
        super(MySageConv, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.root_weight = root_weight
        self.lin_l = torch.nn.Linear(in_channels, hidden_channels, bias=bias)
        if self.root_weight:
            self.lin_r = torch.nn.Linear(in_channels,
                                         hidden_channels,
                                         bias=False)
        self.mean_forward = mean_forward
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x, ptr, idx, num_node):
        if self.mean_forward:
            out = sage_mean_forward(x, ptr, idx, num_node)
        else:
            out = x[:num_node]
        out = self.lin_l(out)
        if self.root_weight:
            if (num_node != 0):
                out += self.lin_r(x[:num_node])
            else:
                out += self.lin_r(x)
        return out


class MyGINConv(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels):
        super(MyGINConv, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.nn = torch.nn.Linear(in_channels, hidden_channels, bias=False)
        self.init_eps = 0.2
        self.eps = torch.nn.Parameter(torch.Tensor([self.init_eps]))
        self.reset_parameters()

    def reset_parameters(self):
        self.nn.reset_parameters()
        self.eps.data.fill_(self.init_eps)

    def forward(self, x, ptr, idx, num_node):
        out = sage_mean_forward(x, ptr, idx, num_node)
        out += (1 + self.eps) * x[:num_node]
        out = self.nn(out)
        return out


class RGCNOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weights, ptr, idx, rel, num_center):
        ctx.save_for_backward(x, weights, ptr, idx, rel)
        num_rel = weights.shape[0]
        output = torch.zeros([num_center, weights.shape[-1]], device=x.device)
        for i in range(num_rel):
            transformed_x = torch.mm(x, weights[i])
            selective_aggr(transformed_x, ptr, idx, (rel == i), output,
                           num_center)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        x, weights, ptr, idx, rel = ctx.saved_tensors
        num_rel = weights.shape[0]
        grad_x = torch.zeros_like(x)
        grad_weights = []
        num_center = grad_out.shape[0]
        num_node = x.shape[0]
        x_t = x.transpose(0, 1)
        for i in range(num_rel):
            grad_selective = torch.zeros([num_node, grad_out.shape[-1]],
                                         device=x.device)
            selective_aggr_bwd(grad_out, ptr, idx, (rel == i), grad_selective,
                               num_center)  # pass grad through selective_aggr
            grad_x += torch.mm(grad_selective, weights[i].transpose(0, 1))
            grad_weights.append(torch.mm(x_t, grad_selective))
        return grad_x, torch.stack(grad_weights), None, None, None, None


class RGCNOP2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weights, ptr, idx, rel, num_center):
        num_rel = weights.shape[0]
        output = torch.zeros([num_center, weights.shape[-1]], device=x.device)
        aggr_outputs = []
        for i in range(num_rel):
            aggr_output = torch.zeros([num_center, weights.shape[-2]],
                                      device=x.device)
            selective_aggr(x, ptr, idx, (rel == i), aggr_output, num_center)
            output += torch.mm(aggr_output, weights[i])
            # output += torch.empty([num_center, weights.shape[-1]],
            #                       device=x.device)
            aggr_outputs.append(aggr_output)
        ctx.save_for_backward(x, weights, ptr, idx, rel,
                              torch.stack(aggr_outputs))
        return output

    @staticmethod
    def backward(ctx, grad_out):
        x, weights, ptr, idx, rel, aggr_outputs = ctx.saved_tensors
        num_rel = weights.shape[0]
        grad_x = torch.zeros_like(x)
        grad_weights = []
        num_center = grad_out.shape[0]
        for i in range(num_rel):
            grad_mm = torch.mm(grad_out, weights[i].transpose(0, 1))
            # grad_mm = torch.empty([num_center, weights.shape[-2]],
            #                       device=x.device)
            grad_weights.append(
                torch.mm(aggr_outputs[i].transpose(0, 1), grad_out))
            selective_aggr_bwd(grad_mm, ptr, idx, (rel == i), grad_x,
                               num_center)  # pass grad through selective_aggr
        return grad_x, torch.stack(grad_weights), None, None, None, None


class RGCNOP3(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weights, ptr, idx, rel, num_center):
        ctx.save_for_backward(x, weights, ptr, idx, rel)
        output = aggr_rgcn_direct_func(x, ptr, idx, weights, rel.int(),
                                       num_center)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        x, weights, ptr, idx, rel = ctx.saved_tensors
        return torch.randn_like(x), torch.randn_like(
            weights), None, None, None, None


class MyRGCNConv(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, num_rel):
        super(MyRGCNConv, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_rel = num_rel
        self.linear = torch.nn.Parameter(
            torch.randn(num_rel, in_channels, hidden_channels))
        self.register_parameter("rel_weight", self.linear)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.linear)
        # self.linear.reset_parameters()
        pass

    def forward(self, x, ptr, idx, edge_types, num_node):
        out = RGCNOP2.apply(x, self.linear, ptr, idx, edge_types, num_node)
        return out


class MyRGCNConvNaive(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, num_rel):
        super(MyRGCNConvNaive, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_rel = num_rel
        self.linear = torch.nn.Parameter(
            torch.randn(num_rel, in_channels, hidden_channels))
        self.register_parameter("rel_weight", self.linear)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()
        pass

    def forward(self, x, edge_index, rel, num_node, num_edge):

        assert (num_edge != 0)
        scattered_feature = torch.index_select(x,
                                               dim=0,
                                               index=edge_index[0][:num_edge])
        scattered_weight = torch.index_select(self.linear,
                                              dim=0,
                                              index=rel[:num_edge])
        transformed_feat = torch.mm(scattered_feature, scattered_weight)
        # out = sage_mean_forward(
        #     transformed_feat, ptr, idx, num_node)
        out = gather(transformed_feat, edge_index[1][:num_edge], num_node)
        return out


class MyRGCNConvOpt1(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, num_rel):
        super(MyRGCNConvOpt1, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_rel = num_rel
        self.linear = torch.nn.Parameter(
            torch.randn(self.num_rel, in_channels, hidden_channels))
        self.register_parameter("rel_weight", self.linear)
        self.reset_parameters()

    def reset_parameters(self):
        # self.linear.reset_parameters()
        pass

    def forward(self, x, ptr, idx, rel, num_node, num_used_node):
        x = x[:num_used_node]
        transformed_feat = torch.matmul(x, self.linear)
        transformed_feat = transformed_feat.reshape(-1,
                                                    transformed_feat.shape[-1])
        out = aggr_rel(transformed_feat, ptr, idx, rel, num_node, self.num_rel)
        return out


class MyRGCNConvOpt2(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, num_rel):
        super(MyRGCNConvOpt2, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_rel = num_rel
        self.linear = torch.nn.Parameter(
            torch.randn(num_rel, in_channels, hidden_channels))
        self.register_parameter("rel_weight", self.linear)
        self.reset_parameters()

    def reset_parameters(self):
        # self.linear.reset_parameters()
        pass

    def forward(self, x, etype_partition, typed_num_node_in_layer, num_node,
                layer_id, num_layer):
        arr = []
        arr_target = []
        # print("overall", typed_num_node_in_layer)
        for i in range(self.num_rel):
            typed_num_node = typed_num_node_in_layer[num_layer * i +
                                                     (num_layer - 1 -
                                                      layer_id)]
            sub_ptr = etype_partition[3 * i]
            sub_idx = etype_partition[3 * i + 1]
            sub_target = etype_partition[3 * i + 2][:typed_num_node]
            # print(typed_num_node, sub_ptr.shape)
            # torch.cuda.synchronize()
            out = sage_sum_forward(x, sub_ptr, sub_idx, typed_num_node)
            # torch.cuda.synchronize()
            out = torch.matmul(out, self.linear[i])
            # torch.cuda.synchronize()
            arr.append(out)
            arr_target.append(sub_target)
        arr = torch.concat(arr, 0)
        arr_target = torch.concat(arr_target, 0)
        out = gather(arr, arr_target, num_node)
        return out


class MyGATConv(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 heads: int = 1,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # if out_channels % heads != 0:
        #     heads = 1
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_src = Parameter(
            torch.Tensor(in_channels, heads * out_channels))

        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_src)
        glorot(self.att_src)
        glorot(self.att_dst)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward_many(self, x, ptr, idx, num_dst, num_src, num_edge):
        H, C = self.heads, self.out_channels
        assert x.dim() == 2
        x_src = x_dst = torch.mm(x[:num_src], self.lin_src).view(-1, H, C)
        alpha_src = (x_src * self.att_src).sum(dim=-1).view(-1, H)
        alpha_dst = (x_dst[:num_dst] * self.att_dst).sum(dim=-1).view(-1, H)
        edge_value = edge_attention(ptr=ptr,
                                    idx=idx,
                                    att_src=alpha_src,
                                    att_dst=alpha_dst,
                                    num_edge=num_edge,
                                    relu_l=self.negative_slope)
        out = sage_sum_forward_edge_value(x_src, ptr, idx, edge_value, num_dst)
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)  # NOTE: requires out to be [-1, H, C]
        if self.bias is not None:
            out += self.bias
        return out

    def forward_1(self, x, ptr, idx, num_dst, num_src, num_edge):
        alpha_src = torch.einsum(
            "mn,nho,ho->mh", x,
            self.lin_src.view(-1, self.heads, self.out_channels),
            self.att_src.squeeze(0)).view(-1, self.heads)
        alpha_dst = torch.einsum(
            "mn,nho,ho->mh", x[:num_dst],
            self.lin_src.view(-1, self.heads, self.out_channels),
            self.att_dst.squeeze(0)).view(-1, self.heads)
        edge_value = edge_attention(ptr=ptr,
                                    idx=idx,
                                    att_src=alpha_src,
                                    att_dst=alpha_dst,
                                    num_edge=num_edge,
                                    relu_l=self.negative_slope)
        if self.out_channels < self.in_channels:
            transformed = torch.mm(x,
                                   self.lin_src).view(-1, self.heads,
                                                      self.out_channels)
            x = sage_sum_forward_edge_value(transformed, ptr, idx, edge_value,
                                            num_dst).squeeze()
        else:
            x = sage_sum_forward_edge_value(x, ptr, idx, edge_value.squeeze(),
                                            num_dst)
            x = torch.mm(x, self.lin_src)
        if self.bias is not None:
            x += self.bias
        return x

    def forward(self, x, ptr, idx, num_dst, num_src, num_edge):
        # return self.forward_many(x, ptr, idx, num_dst, num_src, num_edge)
        if self.heads == 1:
            return self.forward_1(x, ptr, idx, num_dst, num_src, num_edge)
        else:
            return self.forward_many(x, ptr, idx, num_dst, num_src, num_edge)


def gcn_norm(ptr: torch.Tensor, idx: torch.Tensor):
    deg = ptr[1:] - ptr[:-1]  # in degree
    # deg_from = idx.bincount()
    # deg_from = deg_from.index_select(0, idx)
    deg_to = deg.repeat_interleave(deg)
    # assert(deg_to.shape == deg_from.shape)
    # edge_value = (deg_to * deg_from).pow(-1/2)
    edge_value = (deg_to.float()).pow(-1)
    edge_value.masked_fill_(edge_value == float('inf'), 0.)
    return edge_value


class MyGCNConv(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 normalize: bool = True,
                 bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin = Linear(
            in_channels, out_channels, bias=False,
            weight_initializer='glorot')  # for consistency with PyG
        # self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x, ptr, idx, num_node):
        out = self.lin(x)  # order of lin and aggregation is consistent to PyG
        if self.normalize:
            edge_value = gcn_norm(ptr, idx)
            out = sage_sum_forward_edge_value(out, ptr, idx, edge_value,
                                              num_node)
        else:
            out = sage_sum_forward(x, ptr, idx, num_node)

        if self.bias is not None:
            out += self.bias
        return out
