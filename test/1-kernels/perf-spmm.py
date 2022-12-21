import torch
import pytest
import cxgnndl
import cxgnncomp as cxgc
import cxgnncomp_backend
import cxgnndl_backend
from hydra import initialize, compose
from cxgnncomp.util import log
import hydra


# def test_perf_spmm_full_graph():
#     dsets = ["arxiv", "products"]
#     for dset in dsets:
#         ptr, idx = cxgnndl.load_full_graph_structure(dset)
#         ptr = torch.from_numpy(ptr).cuda()
#         idx = torch.from_numpy(idx).cuda()
#         num_all_nodes = ptr.shape[0] - 1
#         for feat_len in [16, 32, 64, 128, 256, 512]:
#             if num_all_nodes * feat_len * 4 < 10 * (1024**3):
#                 # test full graph
#                 x = torch.randn(num_all_nodes, feat_len,
#                                 dtype=torch.float32, device='cuda')
#                 cxgc.prof(f"full {dset} spmm-{feat_len}", "triton",
#                           lambda: cxgc.spmm_triton(x, ptr, idx, num_all_nodes))
#                 cxgc.prof(f"full {dset} spmm-{feat_len}", "manual",
#                           lambda: cxgc.sage_sum_forward(x, ptr, idx, num_all_nodes))
#                 cxgc.prof(f"full {dset} spmm-{feat_len}", "torch",
#                           lambda: cxgc.codegen.torch_spmm.spmm_torch(x, ptr, idx, num_all_nodes))
#             else:
#                 print(f"{dset} {feat_len} too large, skip full graph")


def test_perf_spmm_sampled_graph():
    # dsets = ["papers100M"]
    dsets = ["arxiv", "products", "papers100M",
             "mag240m", "rmag240m", "twitter", "friendster"]
    fanouts = [20, 15, 10]
    for dset in dsets:
        for num_seeds in range(1000, 20000, 2000):
            full_ptr, full_idx = cxgnndl.load_full_graph_structure(dset)
            full_ptr = torch.from_numpy(full_ptr)
            full_idx = torch.from_numpy(full_idx)
            num_all_nodes = full_ptr.shape[0] - 1
            seed_nodes = torch.randint(0, num_all_nodes, (num_seeds,))
            ptr, idx, input_nodes, num_node_in_layer, num_edge_in_layer = cxgnndl_backend.neighbor_sample(
                full_ptr, full_idx, fanouts, seed_nodes)
            ptr = ptr.cuda()
            idx = idx.cuda()
            ptr_for_torch = torch.cat([ptr, torch.ones(
                [input_nodes.shape[0] - ptr.shape[0] + 1], device='cuda', dtype=torch.int64) * ptr[-1].item()], dim=0)
            ptr_for_dgsparse = ptr.to(torch.int32)
            idx_for_dgsparse = idx.to(torch.int32)
            # for feat_len in [512]:
            for feat_len in [256]:
                # for feat_len in [16, 32, 64, 128, 256, 512]:
                # test sampled graph
                try:
                    x = torch.randn(input_nodes.shape[0], feat_len,
                                    dtype=torch.float32, device='cuda')
                except RuntimeError:
                    print(
                        f"{dset} {feat_len} {num_seeds} too large, skip sampled graph")
                    break
                output_triton = cxgc.spmm_triton(x, ptr, idx, ptr.shape[0] - 1)
                output_manual = cxgc.sage_sum_forward(
                    x, ptr, idx, ptr.shape[0] - 1)
                output_dgsparse = cxgnncomp_backend.GSpMM_u(
                    ptr_for_dgsparse, idx_for_dgsparse, x, cxgnncomp_backend.REDUCEOP.SUM)
                assert torch.allclose(
                    output_triton, output_manual, atol=1e-5, rtol=1e-4)
                assert torch.allclose(
                    output_triton, output_dgsparse, atol=1e-5, rtol=1e-4)

                cxgc.prof(f"sample {num_seeds} {dset} spmm-{feat_len}", "triton",
                          lambda: cxgc.spmm_triton(x, ptr, idx, ptr.shape[0] - 1))
                cxgc.prof(f"sample {num_seeds} {dset} spmm-{feat_len}", "manual",
                          lambda: cxgc.sage_sum_forward(x, ptr, idx, ptr.shape[0] - 1))
                cxgc.prof(f"sample {num_seeds} {dset} spmm-{feat_len}", "dgsparse",
                          lambda: cxgnncomp_backend.GSpMM_u(ptr_for_dgsparse, idx_for_dgsparse, x, cxgnncomp_backend.REDUCEOP.SUM))
                # cxgc.prof(f"sample {num_seeds} {dset} spmm-{feat_len}", "torch",
                #           lambda: cxgc.codegen.torch_spmm.spmm_torch(x, ptr_for_torch, idx, ptr_for_torch.shape[0] - 1))
