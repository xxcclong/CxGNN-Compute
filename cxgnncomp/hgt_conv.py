import torch
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, ones, reset


class MyHGTConv(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_node_type,
        num_edge_type,
        heads,
    ) -> None:
        super(MyHGTConv, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        dim = hidden_channels // heads
        assert (dim * heads == hidden_channels)
        self.k_lin = Parameter(
            torch.Tensor(num_node_type, in_channels, heads, dim))
        self.q_lin = Parameter(
            torch.Tensor(num_node_type, in_channels, heads, dim))
        self.v_lin = Parameter(
            torch.Tensor(num_node_type, in_channels, heads, dim))

        self.a_rel = Parameter(torch.Tensor(num_edge_type, heads, dim, dim))
        self.m_rel = Parameter(torch.Tensor(num_edge_type, heads, dim, dim))
        self.p_rel = Parameter(torch.Tensor(num_edge_type, heads, dim, dim))

        self.relation_ptr = Parameter(torch.Tensor(heads, num_edge_type))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.k_lin)
        reset(self.q_lin)
        reset(self.v_lin)
        reset(self.a_rel)
        reset(self.m_rel)
        reset(self.p_rel)

    def linear_typed(self, x, lin, types):
        return torch.Tensor([])

    def linear_typed_multihead(self, x, lin, types):
        return torch.Tensor([])

    def mul_typed(self, x, y, types):
        return torch.Tensor([])

    def forward(self, x, ptr, idx, node_type, edge_type, num_center, num_src,
                num_edge):
        dst_node_type = node_type[:num_center]
        src_node_type = node_type[:num_src]
        x_dst = x[:num_center]
        x_src = x[:num_src]
        src_id = idx[:num_edge]
        edge_type = edge_type[:num_edge]

        # src data [num_src, heads, dim]
        k = self.linear_typed(
            x_src,
            self.k_lin,
            src_node_type,
        )
        # dst data [num_center, heads, dim]
        q = self.linear_typed(
            x_dst,
            self.q_lin,
            dst_node_type,
        )
        # src data [num_src, heads, dim]
        v = self.linear_typed(
            x_src,
            self.v_lin,
            src_node_type,
        )
        kw = self.linear_typed_multihead(k, self.a_rel,
                                         edge_type)  # [num_edge, heads, dim]
        a = torch.einsum("ehd,ehd->eh", kw, torch.index_select(q, 0, src_id))

        pass