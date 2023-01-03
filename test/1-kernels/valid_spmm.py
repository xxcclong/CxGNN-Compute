import cxgnncomp
import cxgnndl
import cxgnncomp_backend
import torch
import pytest
from torch_scatter import segment_csr, gather_csr
from torch.profiler import profile, record_function, ProfilerActivity

feat_len = 128

def allclose(a,b):
    return torch.allclose(a,b,atol=1e-3,rtol=1e-2)

def prepare_data():
    torch.manual_seed(0)
    # dataset_name = "paper100m"
    # file_dir = "/home/huangkz/repos/new-diskgnn/DiskGNN/graph_loader/{}_batch.pt".format(
    #     dataset_name)
    file_dir = "/home/huangkz/repos/CxGNN-DL/dump.pt"
    batch = torch.load(file_dir)
    x = torch.randn([batch["num_node_in_layer"][-1], feat_len],
                    dtype=torch.float32,
                    device='cuda')
    ptr = batch["ptr"].cuda()
    idx = batch["idx"].cuda()
    return x, ptr, idx, batch, None

def prepare_data_full_graph():
    ptr, idx = cxgnndl.load_full_graph_structure("arxiv")
    ptr = torch.from_numpy(ptr).cuda()
    idx = torch.from_numpy(idx).cuda()
    x = torch.randn([ptr.shape[0] - 1, feat_len],
                    dtype=torch.float32,
                    device='cuda')
    edge_index = torch.stack([
        idx,
        torch.repeat_interleave(torch.arange(
            0, ptr.shape[0] - 1, device="cuda"),
                                repeats=ptr[1:] - ptr[:-1])
    ],
                             dim=0)
    batch = {}
    batch["num_node_in_layer"] = [ptr.shape[0] - 1, ptr.shape[0] - 1]
    return x, ptr, idx, batch, edge_index


x, ptr, idx, batch, edge_index = prepare_data_full_graph()
output_torch = cxgnncomp.codegen.torch_spmm.spmm_torch(
    x, ptr, idx, ptr.shape[0] - 1)


def test_manual_spmm():
    output_manual = cxgnncomp.sage_sum_forward(x, ptr, idx, ptr.shape[0] - 1)
    assert allclose(output_torch, output_manual)
    cxgnncomp.prof("manual", f"spmm-{feat_len}", lambda: cxgnncomp.sage_sum_forward(x, ptr, idx, ptr.shape[0] - 1))
    cxgnncomp.prof("manual", f"spmm-{feat_len}", lambda: cxgnncomp.sage_sum_forward(x, ptr, idx, ptr.shape[0] - 1))


def test_triton_spmm():
    output_triton = cxgnncomp.spmm_triton(x, ptr, idx, ptr.shape[0] - 1)
    assert allclose(output_torch, output_triton)
    cxgnncomp.prof("triton", f"spmm-{feat_len}", lambda: cxgnncomp.spmm_triton(x, ptr, idx, ptr.shape[0] - 1))


def test_dgsparse_spmm():
    ptr_int = ptr.to(torch.int32)
    idx_int = idx.to(torch.int32)
    output_dgsparse = cxgnncomp_backend.GSpMM_u(ptr_int, idx_int, x, cxgnncomp_backend.REDUCEOP.SUM)
    assert allclose(output_torch, output_dgsparse)
    cxgnncomp.prof("dgsparse", f"spmm-{feat_len}", lambda: cxgnncomp_backend.GSpMM_u(ptr_int, idx_int, x, cxgnncomp_backend.REDUCEOP.SUM))


random_value = torch.randn_like(idx, dtype=torch.float32)
output_torch_val = cxgnncomp.codegen.torch_spmm.spmm_torch(
    x, ptr, idx, ptr.shape[0] - 1, random_value)


def test_manual_spmm_val():
    output_manual = cxgnncomp.sage_sum_forward_edge_value(
        x, ptr, idx, random_value, ptr.shape[0] - 1)
    assert allclose(output_torch_val, output_manual)
    cxgnncomp.prof("manual", f"spmm-val-{feat_len}", lambda: cxgnncomp.sage_sum_forward_edge_value(x, ptr, idx, random_value, ptr.shape[0] - 1))


def test_triton_spmm_val():
    output_triton = cxgnncomp.spmm_triton(
        x, ptr, idx, ptr.shape[0] - 1, random_value)
    assert allclose(output_torch_val, output_triton)
    cxgnncomp.prof("triton", f"spmm-val-{feat_len}", lambda: cxgnncomp.spmm_triton(x, ptr, idx, ptr.shape[0] - 1, random_value))

def test_op_spmm():
    expanded = torch.index_select(x, 0, idx)
    output_op = segment_csr(expanded, ptr, reduce="sum")
    assert allclose(output_torch, output_op)
    cxgnncomp.prof("OP", f"spmm-{feat_len}", lambda: segment_csr(torch.index_select(x, 0, idx), ptr, reduce="sum"))

def test_op_spmm_val():
    val = random_value.view(-1, 1)
    expanded = torch.index_select(x, 0, idx)
    output_op = segment_csr(expanded * val, ptr, reduce="sum")
    assert allclose(output_torch_val, output_op)
    cxgnncomp.prof("OP", f"spmm-val-{feat_len}", lambda: segment_csr(torch.index_select(x, 0, idx) * val, ptr, reduce="sum"))

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #              record_shapes=True) as prof:
    #     expanded = torch.index_select(x, 0, idx)
    #     segment_csr(expanded, ptr, reduce="sum")
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))