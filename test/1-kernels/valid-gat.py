import cxgnncomp
import cxgnncomp_backend
import torch
import pytest
import torch_geometric
import dgl
from copy import deepcopy
import cxgnndl
from torch.profiler import profile, record_function, ProfilerActivity

feat_len = 128

torch.manual_seed(0)

x, ptr, idx, batch, edge_index = cxgnncomp.prepare_data_full_graph(
    need_edge_index=True)
num_node_in_layer = batch["num_node_in_layer"]


def test_cxgnn_gat():
    pyg_conv = torch_geometric.nn.GATConv(
        in_channels=feat_len,
        out_channels=feat_len,
        add_self_loops=False,
    ).cuda()
    pyg_conv.reset_parameters()

    cxgnn_conv = cxgnncomp.MyGATConv(
        in_channels=feat_len,
        out_channels=feat_len,
    ).cuda()

    dgl_conv = dgl.nn.GATConv(
        in_feats=feat_len,
        out_feats=feat_len,
        num_heads=1,
    ).cuda()

    cxgnn_conv.lin_src = deepcopy(pyg_conv.lin_src.weight)
    cxgnn_conv.att_src = deepcopy(pyg_conv.att_src)
    cxgnn_conv.att_dst = deepcopy(pyg_conv.att_dst)
    cxgnn_conv.bias = deepcopy(pyg_conv.bias)

    dgl_conv.fc = deepcopy(pyg_conv.lin_src)
    dgl_conv.attn_l = deepcopy(pyg_conv.att_src)
    dgl_conv.attn_r = deepcopy(pyg_conv.att_dst)
    dgl_conv.bias = deepcopy(pyg_conv.bias)

    pyg_out = pyg_conv(x, edge_index)
    cxgnn_out = cxgnn_conv(x, ptr, idx, num_node_in_layer[-2],
                           num_node_in_layer[-1], idx.shape[0])
    dgl_graph = dgl.DGLGraph((edge_index[0], edge_index[1]))
    dgl_out = dgl_conv(dgl_graph, x).squeeze()

    # pos = ~torch.isclose(pyg_out, cxgnn_out, atol=1e-2, rtol=1e-2)
    # print(torch.sum(pos), pos.shape)
    # print((pyg_out - cxgnn_out)[pos], cxgnn_out[pos])
    print(pyg_out.shape, dgl_out.shape, cxgnn_out.shape)

    assert torch.allclose(pyg_out, cxgnn_out, atol=1e-2, rtol=1e-2)
    assert torch.allclose(pyg_out, dgl_out, atol=1e-2, rtol=1e-2)

    cxgnncomp.prof("cxgnn_gat_single_head.prof", "dgl",
                   lambda: dgl_conv(dgl_graph, x).squeeze())

    cxgnncomp.prof("cxgnn_gat_single_head.prof", "pyg",
                   lambda: pyg_conv(x, edge_index))

    cxgnncomp.prof(
        "cxgnn_gat_single_head.prof", "cxgnn",
        lambda: cxgnn_conv(x, ptr, idx, num_node_in_layer[-2],
                           num_node_in_layer[-1], idx.shape[0]))


def test_cxgnn_gat_multi_head():
    num_head = 4
    pyg_conv = torch_geometric.nn.GATConv(in_channels=feat_len,
                                          out_channels=feat_len,
                                          add_self_loops=False,
                                          heads=num_head).cuda()
    pyg_conv.reset_parameters()

    cxgnn_conv = cxgnncomp.MyGATConv(in_channels=feat_len,
                                     out_channels=feat_len,
                                     heads=num_head).cuda()

    dgl_conv = dgl.nn.GATConv(
        in_feats=feat_len,
        out_feats=feat_len,
        num_heads=num_head,
    ).cuda()

    cxgnn_conv.lin_src = deepcopy(pyg_conv.lin_src.weight)
    print(cxgnn_conv.lin_src.shape)
    cxgnn_conv.att_src = deepcopy(pyg_conv.att_src)
    cxgnn_conv.att_dst = deepcopy(pyg_conv.att_dst)
    cxgnn_conv.bias = deepcopy(pyg_conv.bias)

    dgl_conv.fc = deepcopy(pyg_conv.lin_src)
    dgl_conv.attn_l = deepcopy(pyg_conv.att_src)
    dgl_conv.attn_r = deepcopy(pyg_conv.att_dst)
    dgl_conv.bias = deepcopy(pyg_conv.bias)

    pyg_out = pyg_conv(x, edge_index)
    cxgnn_out = cxgnn_conv(x, ptr, idx, num_node_in_layer[-2],
                           num_node_in_layer[-1], idx.shape[0])
    dgl_graph = dgl.DGLGraph((edge_index[0], edge_index[1]))
    dgl_out = dgl_conv(dgl_graph, x).squeeze()

    assert torch.allclose(pyg_out, cxgnn_out, atol=1e-4, rtol=1e-3)
    assert torch.allclose(pyg_out,
                          dgl_out.view(-1, num_head * feat_len),
                          atol=1e-2,
                          rtol=1e-2)

    cxgnncomp.prof("cxgnn_gat_multi_head.prof", "dgl",
                   lambda: dgl_conv(dgl_graph, x).squeeze())

    cxgnncomp.prof("cxgnn_gat_multi_head.prof", "pyg",
                   lambda: pyg_conv(x, edge_index))

    cxgnncomp.prof(
        "cxgnn_gat_multi_head.prof", "cxgnn",
        lambda: cxgnn_conv(x, ptr, idx, num_node_in_layer[-2],
                           num_node_in_layer[-1], idx.shape[0]))


def test_edge_softmax():
    from torch_geometric.utils import softmax
    num_head = 1
    att = torch.randn([idx.shape[0], num_head], device="cuda")
    att_src = torch.randn([x.shape[0], num_head], device="cuda")
    att_dst = torch.randn([x.shape[0], num_head], device="cuda")

    cxgnn_conv = cxgnncomp.MyGATConv(in_channels=feat_len,
                                     out_channels=feat_len,
                                     heads=num_head).cuda()
    cxgnncomp.prof("edge_softmax", "pyg", lambda: softmax(att, edge_index[0]))
    cxgnncomp.prof(
        "edge_softmax", "cxgnn fused",
        lambda: cxgnn_conv.edge_softmax_fused(ptr=ptr,
                                              idx=idx,
                                              att_src=att_src,
                                              att_dst=att_dst,
                                              num_edge=idx.shape[0],
                                              relu_l=0.2))
    cxgnncomp.prof(
        "edge_softmax", "cxgnn opwise",
        lambda: cxgnn_conv.edge_softmax_opwise(ptr=ptr,
                                               idx=idx,
                                               att_src=att_src,
                                               att_dst=att_dst,
                                               num_edge=idx.shape[0],
                                               relu_l=0.2))

    cxgnncomp.prof(
        "edge_softmax", "spmv", lambda: cxgnncomp_backend.spmv(
            ptr, idx, att_src.squeeze(), ptr.shape[0] - 1))

    for thres in [32, 64, 128, 256, 512]:
        new_ptr, new_target = cxgnncomp.neighbor_grouping(ptr,
                                                          neighbor_thres=thres)
        cxgnncomp.prof(
            "edge_softmax",
            "spmv-neighbor-group", lambda: cxgnncomp_backend.spmv(
                new_ptr, idx, att_src.squeeze(), new_ptr.shape[0] - 1))


def test_edge_softmax_multi_head():
    from torch_geometric.utils import softmax
    num_head = 4
    print(f"test_edge_softmax_multi_head head={num_head}")
    att = torch.randn([idx.shape[0], num_head], device="cuda")
    att_src = torch.randn([x.shape[0], num_head], device="cuda")
    att_dst = torch.randn([x.shape[0], num_head], device="cuda")

    cxgnn_conv = cxgnncomp.MyGATConv(in_channels=feat_len,
                                     out_channels=feat_len,
                                     heads=num_head).cuda()
    # cxgnncomp.prof("edge_softmax", "pyg", lambda: softmax(att, edge_index[0]))
    cxgnncomp.prof(
        "edge_softmax", "cxgnn fused",
        lambda: cxgnn_conv.edge_softmax_fused(ptr=ptr,
                                              idx=idx,
                                              att_src=att_src,
                                              att_dst=att_dst,
                                              num_edge=idx.shape[0],
                                              relu_l=0.2))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        cxgnncomp.prof(
            "edge_softmax", "cxgnn opwise",
            lambda: cxgnn_conv.edge_softmax_opwise(ptr=ptr,
                                                   idx=idx,
                                                   att_src=att_src,
                                                   att_dst=att_dst,
                                                   num_edge=idx.shape[0],
                                                   relu_l=0.2))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def test_gat_timer():
    num_head = 4
    cxgnn_conv = cxgnncomp.MyGATConv(in_channels=feat_len,
                                     out_channels=feat_len,
                                     heads=num_head).cuda()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        cxgnn_out = cxgnn_conv(x, ptr, idx, num_node_in_layer[-2],
                               num_node_in_layer[-1], idx.shape[0])
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    pyg_conv = torch_geometric.nn.GATConv(in_channels=feat_len,
                                          out_channels=feat_len,
                                          add_self_loops=False,
                                          heads=num_head).cuda()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        pyg_out = pyg_conv(x, edge_index)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    dgl_conv = dgl.nn.GATConv(
        in_feats=feat_len,
        out_feats=feat_len,
        num_heads=num_head,
    ).cuda()

    dgl_graph = dgl.DGLGraph((edge_index[0], edge_index[1]))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        dgl_out = dgl_conv(dgl_graph, x).squeeze()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


test_edge_softmax()
test_edge_softmax_multi_head()