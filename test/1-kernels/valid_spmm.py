import cxgnncomp
import cxgnncomp_backend
import torch
import pytest


def prepare_data():
    torch.manual_seed(0)
    # dataset_name = "paper100m"
    # file_dir = "/home/huangkz/repos/new-diskgnn/DiskGNN/graph_loader/{}_batch.pt".format(
    #     dataset_name)
    file_dir = "/home/huangkz/repos/CxGNN-DL/dump.pt"
    batch = torch.load(file_dir)
    feat_len = 128
    x = torch.randn([batch["num_node_in_layer"][-1], feat_len],
                    dtype=torch.float32,
                    device='cuda')
    ptr = batch["ptr"].cuda()
    idx = batch["idx"].cuda()
    return x, ptr, idx, batch


x, ptr, idx, batch = prepare_data()
output_torch = cxgnncomp.codegen.torch_spmm.spmm_torch(
    x, ptr, idx, ptr.shape[0] - 1)


def test_manual_spmm():
    output_manual = cxgnncomp.sage_sum_forward(x, ptr, idx, ptr.shape[0] - 1)
    assert torch.allclose(output_torch, output_manual, atol=1e-5)


def test_triton_spmm():
    output_triton = cxgnncomp.spmm_triton(x, ptr, idx, ptr.shape[0] - 1)
    assert torch.allclose(output_torch, output_triton, atol=1e-5)


def test_dgsparse_spmm():
    output_dgsparse = cxgnncomp_backend.GSpMM_u(
        ptr.to(torch.int32), idx.to(torch.int32), x, cxgnncomp_backend.REDUCEOP.SUM)
    assert torch.allclose(output_torch, output_dgsparse, atol=1e-5)


random_value = torch.randn_like(idx, dtype=torch.float32)
output_torch_val = cxgnncomp.codegen.torch_spmm.spmm_torch(
    x, ptr, idx, ptr.shape[0] - 1, random_value)


def test_manual_spmm_val():
    output_manual = cxgnncomp.sage_sum_forward_edge_value(
        x, ptr, idx, random_value, ptr.shape[0] - 1)
    assert torch.allclose(output_torch_val, output_manual,
                          rtol=1e-4, atol=1e-5)


def test_triton_spmm_val():
    output_triton = cxgnncomp.spmm_triton(
        x, ptr, idx, ptr.shape[0] - 1, random_value)
    assert torch.allclose(output_torch_val, output_triton,
                          atol=1e-5, rtol=1e-4)


def test_triton_spmm_val():
    output_triton = cxgnncomp.spmm_triton(
        x, ptr, idx, ptr.shape[0] - 1, random_value)
    assert torch.allclose(output_torch_val, output_triton,
                          atol=1e-5, rtol=1e-4)
