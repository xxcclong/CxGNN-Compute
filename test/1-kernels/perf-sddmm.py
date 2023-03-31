import cxgnncomp as cxgc
import cxgnncomp_backend
import time
import torch


def test_sddmm():
    infeat = 256
    num_head = 4
    dev = torch.device("cuda:0")

    # dset = "arxiv"
    # feat, ptr, idx, b, edge_index = cxgc.prepare_data_full_graph(
    #     dset, feat_len=infeat, num_head=num_head, need_edge_index=True)

    dset = "papers100M"
    feat, ptr, idx, b, edge_index = cxgc.prepare_data_sampled_graph(
        dset=dset,
        feat_len=infeat,
        num_head=num_head,
        num_seeds=1000,
        need_edge_index=True)

    # feat = torch.ones_like(feat)
    # dst_feat = torch.ones([ptr.shape[0] - 1, infeat], device=feat.device)
    dst_feat = torch.randn([ptr.shape[0] - 1, infeat], device=feat.device)
    output = torch.zeros([idx.shape[0], num_head], device=feat.device)

    cxgc.set_timers()
    cxgnncomp_backend.run_sddmm(edge_index[0], edge_index[1], feat, dst_feat,
                                output, edge_index.shape[1])

    output_time = cxgc.prof(
        "sddmm", "edge parallel",
        lambda: cxgnncomp_backend.run_sddmm(edge_index[0], edge_index[
            1], feat, dst_feat, output, edge_index.shape[1]))
    print("sddmm flops: ",
          infeat * num_head * edge_index.shape[1] / output_time[1] / 1e9)

    tuner = cxgc.Tuner()
    print(ptr.shape, idx.shape, feat.shape, dst_feat.shape)
    output2 = tuner.tune_graph(
        ptr.shape[0] - 1,
        idx.shape[0],
        feat.shape[-1],
        cxgnncomp_backend.run_sddmm_vertex_centric,
        [ptr, idx, feat, dst_feat, ptr.shape[0] - 1],
    )

    output = torch.zeros([idx.shape[0], num_head], device=feat.device)
    cxgnncomp_backend.run_sddmm(edge_index[0], edge_index[1], feat, dst_feat,
                                output, edge_index.shape[1])
    print(
        "correct rate output vs output2:",
        torch.sum(
            torch.isclose(
                output.view(-1), output2.view(-1), atol=1e-2, rtol=1e-2)) /
        torch.numel(output))
    print(output)
    print(output2)

    output_time = cxgc.prof(
        "sddmm", "vertex parallel tuned", lambda: tuner.tune_graph(
            ptr.shape[0] - 1,
            idx.shape[0],
            feat.shape[-1],
            cxgnncomp_backend.run_sddmm_vertex_centric,
            [ptr, idx, feat, dst_feat, ptr.shape[0] - 1],
        ))

    print("sddmm vertex-centric flops: ",
          infeat * num_head * edge_index.shape[1] / output_time[1] / 1e9)


if __name__ == "__main__":
    test_sddmm()