import torch
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot
from cxgnncomp_backend import edge_attention, sage_sum_forward_edge_value, gather, sage_sum_forward, aggr_rel, sage_mean_forward, selective_aggr, selective_aggr_bwd, aggr_rgcn_direct_func
from .util import log
import torch.nn.functional as F
from torch_scatter import segment_csr, gather_csr
from .timer import TimerOP
from .graph_kernel import SpMMValOP

torch.fx.wrap("edge_attention")
torch.fx.wrap("sage_sum_forward_edge_value")
torch.fx.wrap("gather")
torch.fx.wrap("sage_sum_forward")
torch.fx.wrap("aggr_rel")
torch.fx.wrap("sage_mean_forward")


class MySageConv(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        root_weight: bool = True,
        bias: bool = True,
    ):
        super(MySageConv, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.root_weight = root_weight
        self.lin_l = torch.nn.Linear(in_channels, hidden_channels, bias=bias)
        if self.root_weight:
            self.lin_r = torch.nn.Linear(in_channels,
                                         hidden_channels,
                                         bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x, ptr, idx, num_node):
        out = sage_mean_forward(x, ptr, idx, num_node)
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


def RGCNOP_sorted(x, weights, src, dst, num_feat_per_rel, num_center):
    num_rel = weights.shape[0]
    output = torch.zeros([num_center, weights.shape[-1]], device=x.device)
    cnt = 0
    for i in range(num_rel):
        s = src[cnt:cnt + num_feat_per_rel[i]]
        d = dst[cnt:cnt + num_feat_per_rel[i]]
        cnt += num_feat_per_rel[i]
        feat = x[s]
        transformed_feat = F.linear(feat, weights[i].T)
        output.index_add_(0, d, transformed_feat)
    return output


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
        log.info("linear shape: {}".format(self.linear.shape))
        self.register_parameter("rel_weight", self.linear)
        self.single_linear = torch.nn.Linear(in_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.linear)
        # self.linear.reset_parameters()
        self.single_linear.reset_parameters()
        pass

    def forward(self, x, ptr, idx, edge_types, num_node):
        out = RGCNOP.apply(x, self.linear, ptr, idx, edge_types, num_node)
        deg = ptr[1:] - ptr[:-1]
        out = out / deg.unsqueeze(-1)[:out.shape[0]]
        # out = self.single_linear(x)
        # out = sage_mean_forward(out, ptr, idx, num_node)
        return out


# this implementation is memory inefficient, it expands the weight parameters from [R, M, N] into [E, M, N] to perform BMM
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
                 concat: bool = False,
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
            torch.Tensor(heads * out_channels, in_channels))

        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.edge_softmax_schedule = "fused"
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_src)
        glorot(self.att_src)
        glorot(self.att_dst)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def edge_softmax_fused(self, ptr, idx, att_src, att_dst, num_edge, relu_l):
        return edge_attention(ptr=ptr,
                              idx=idx,
                              att_src=att_src,
                              att_dst=att_dst,
                              num_edge=num_edge,
                              relu_l=relu_l)

    def edge_softmax_opwise(self, ptr, idx, att_src, att_dst, num_edge,
                            relu_l):
        alpha_src = torch.index_select(att_src, 0, idx[:num_edge])
        alpha_dst = gather_csr(att_dst, ptr)
        alpha = F.leaky_relu(alpha_src + alpha_dst, relu_l)
        with torch.no_grad():
            alpha_max = segment_csr(alpha, ptr, reduce='max')
            alpha_max = gather_csr(alpha_max, ptr)
        alpha = torch.exp(alpha - alpha_max)
        out_sum = segment_csr(alpha, ptr, reduce='sum') + 1e-16
        out_sum = gather_csr(out_sum, ptr)
        edge_value = alpha / out_sum
        return edge_value

    def edge_softmax(self, ptr, idx, att_src, att_dst, num_edge, relu_l):
        if self.edge_softmax_schedule == "fused":
            return self.edge_softmax_fused(ptr, idx, att_src, att_dst,
                                           num_edge, relu_l)
        else:
            return self.edge_softmax_opwise(ptr, idx, att_src, att_dst,
                                            num_edge, relu_l)

    def forward_many(self, x, ptr, idx, num_dst, num_src, num_edge):
        H, C = self.heads, self.out_channels
        assert x.dim() == 2
        # x_src = x_dst = torch.mm(x[:num_src], self.lin_src).view(-1, H, C)
        x = TimerOP.apply(x, "linear1", True)
        x_src = x_dst = F.linear(x[:num_src], self.lin_src).view(-1, H, C)
        x_src = TimerOP.apply(x_src, "linear1", False)
        x_src = TimerOP.apply(x_src, "sum1", True)
        alpha_src = (x_src * self.att_src).sum(dim=-1).view(-1, H)
        alpha_dst = (x_dst[:num_dst] * self.att_dst).sum(dim=-1).view(-1, H)
        alpha_dst = TimerOP.apply(alpha_dst, "sum1", False)
        alpha_dst = TimerOP.apply(alpha_dst, "softmax1", True)
        edge_value = self.edge_softmax(ptr=ptr,
                                       idx=idx,
                                       att_src=alpha_src,
                                       att_dst=alpha_dst,
                                       num_edge=num_edge,
                                       relu_l=self.negative_slope)
        edge_value = TimerOP.apply(edge_value, "softmax1", False)
        edge_value = TimerOP.apply(edge_value, "aggregation", True)
        # out = sage_sum_forward_edge_value(x_src, ptr, idx, edge_value, num_dst)
        out = SpMMValOP.apply(x_src, ptr, idx, edge_value, num_dst)
        out = TimerOP.apply(out, "aggregation", False)
        if self.concat:
            out = out.view(-1, H * C)
        elif out.shape[1] == self.heads:
            out = out.mean(dim=1)  # NOTE: requires out to be [-1, H, C]
        # else: already sumed/averaged
        if self.bias is not None:
            return out + self.bias
        else:
            return out

    def forward_1(self, x, ptr, idx, num_dst, num_src, num_edge):
        x = TimerOP.apply(x, "einsum1", True)
        alpha_src = torch.einsum(
            "mn,nho,ho->mh", x,
            self.lin_src.T.view(-1, self.heads, self.out_channels),
            self.att_src.squeeze(0)).view(-1, self.heads)
        alpha_dst = torch.einsum(
            "mn,nho,ho->mh", x[:num_dst],
            self.lin_src.T.view(-1, self.heads, self.out_channels),
            self.att_dst.squeeze(0)).view(-1, self.heads)
        alpha_dst = TimerOP.apply(alpha_dst, "einsum1", False)
        alpha_dst = TimerOP.apply(alpha_dst, "softmax1", True)
        edge_value = self.edge_softmax(ptr=ptr,
                                       idx=idx,
                                       att_src=alpha_src,
                                       att_dst=alpha_dst,
                                       num_edge=num_edge,
                                       relu_l=self.negative_slope)
        edge_value = TimerOP.apply(edge_value, "softmax1", False)
        if self.out_channels < self.in_channels:
            x = TimerOP.apply(x, "mm1", True)
            transformed = torch.mm(x, self.lin_src.T).view(
                -1, self.heads, self.out_channels)
            transformed = TimerOP.apply(transformed, "mm1", False)
            transformed = TimerOP.apply(transformed, "aggregation", True)
            x = sage_sum_forward_edge_value(transformed, ptr, idx, edge_value,
                                            num_dst).squeeze()
            x = TimerOP.apply(x, "aggregation", False)
        else:
            x = TimerOP.apply(x, "aggregation", True)
            x = sage_sum_forward_edge_value(x, ptr, idx, edge_value.squeeze(),
                                            num_dst)
            x = TimerOP.apply(x, "aggregation", False)
            x = TimerOP.apply(x, "mm1", True)
            x = torch.mm(x, self.lin_src.T)
            x = TimerOP.apply(x, "mm1", False)
        if self.bias is not None:
            # x += self.bias
            return x + self.bias
        else:
            return x

    def forward(self, x, ptr, idx, num_dst, num_src, num_edge):
        # print(x.shape, ptr.shape, idx.shape, num_dst, num_src, num_edge)
        if self.heads == 1:
            return self.forward_1(x, ptr, idx, num_dst, num_src, num_edge)
        else:
            return self.forward_many(x, ptr, idx, num_dst, num_src, num_edge)


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

    def gcn_norm(self, ptr: torch.Tensor, idx: torch.Tensor):
        deg = ptr[1:] - ptr[:-1]  # in degree
        # deg_from = idx.bincount()
        # deg_from = deg_from.index_select(0, idx)
        deg_to = deg.repeat_interleave(deg)
        # assert(deg_to.shape == deg_from.shape)
        # edge_value = (deg_to * deg_from).pow(-1/2)
        edge_value = (deg_to.float()).pow(-1)
        edge_value.masked_fill_(edge_value == float('inf'), 0.)
        return edge_value

    def forward(self, x, ptr, idx, num_node):
        x = TimerOP.apply(x, "linear", True)
        out = self.lin(x)  # order of lin and aggregation is consistent to PyG
        out = TimerOP.apply(out, "linear", False)
        if self.normalize:
            out = TimerOP.apply(out, "aggregation", True)
            TimerOP.apply(out, "gcn norm", True)
            edge_value = self.gcn_norm(ptr, idx)
            TimerOP.apply(edge_value, "gcn norm", False)
            print(out.shape, ptr.shape, idx.shape, edge_value.shape, num_node)
            out = sage_sum_forward_edge_value(out, ptr, idx, edge_value,
                                              num_node)
            out = TimerOP.apply(out, "aggregation", False)
        else:
            out = sage_sum_forward(x, ptr, idx, num_node)

        if self.bias is not None:
            return out + self.bias
        else:
            return out
