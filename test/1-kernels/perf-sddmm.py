import cxgnncomp as cxgc
import cxgnncomp_backend
import time
import torch


def test_sddmm():
    infeat = 256
    num_head = 4
    dev = torch.device("cuda:0")

    dset = "arxiv"
    feat, ptr, idx, b, edge_index = cxgc.prepare_data_full_graph(
        dset, feat_len=infeat, num_head=num_head, need_edge_index=True)

    # dset = "papers100M"
    # feat, ptr, idx, b, edge_index = cxgc.prepare_data_sampled_graph(
    #     dset=dset,
    #     feat_len=infeat,
    #     num_head=num_head,
    #     num_seeds=1000,
    #     need_edge_index=True)

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
          infeat * num_head * edge_index.shape[1] / output_time[1] / 1e9 * 1e3)

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
          infeat * num_head * edge_index.shape[1] / output_time[1] / 1e9 * 1e3)

    ptr, idx = cxgc.partition_2d(
        ptr,
        idx,
        b["num_node_in_layer"][-1],
        ptr.shape[0] - 1,
    )
    dst_feat = torch.randn([ptr.shape[0] - 1, infeat], device=feat.device)
    tuner.tune_graph(
        ptr.shape[0] - 1,
        idx.shape[0],
        feat.shape[-1],
        cxgnncomp_backend.run_sddmm_vertex_centric,
        [ptr, idx, feat, dst_feat, ptr.shape[0] - 1],
    )
    output_time = cxgc.prof(
        "sddmm", "2d-partition tuned", lambda: tuner.tune_graph(
            ptr.shape[0] - 1,
            idx.shape[0],
            feat.shape[-1],
            cxgnncomp_backend.run_sddmm_vertex_centric,
            [ptr, idx, feat, dst_feat, ptr.shape[0] - 1],
        ))
    print("sddmm 2d-partition flops: ",
          infeat * num_head * edge_index.shape[1] / output_time[1] / 1e9 * 1e3)

    num_edge = edge_index.shape[1]
    a = torch.randn([num_edge, infeat], device=dev)
    b = torch.randn([num_edge, num_head, infeat], device=dev)
    output_time = cxgc.prof("sddmm", "dense",
                            lambda: torch.einsum("ij,ihj->ih", a, b))
    print("sddmm dense flops: ",
          infeat * num_edge * num_head / output_time[1] / 1e9 * 1e3)


def test_dense():
    # num_edge = 1500000
    # feat_len = 256
    # num_head = 4
    dev = torch.device("cuda:0")
    for i in [10000]:
        for j in [256]:
            for h in [4]:
                a = torch.randn([i, j], device=dev)
                b = torch.randn([j, 32], device=dev)
                output_time = cxgc.prof("sddmm", "dense",
                                        lambda: torch.mm(a, b))
                # cxgc.prof("sddmm", "add", lambda: torch.add(c, b))
                # cxgc.prof("sddmm", "reduce", lambda: torch.sum(b, dim=2))
                # cxgc.prof("sddmm", "mul", lambda: torch.mul(b, c))
                print(f"sddmm dense flops: {i} {j} {h} ",
                      i * j * 25 / output_time[1] / 1e9 * 1e3)


def test_dense_spmm():
    dev = torch.device("cuda:0")
    num_nodes = 100000
    num_edge_per_node = 32
    feat_len = 256
    a = torch.randn([num_nodes, num_edge_per_node, feat_len], device=dev)
    output_time = cxgc.prof("add", "dense", lambda: torch.sum(a, dim=1))
    print("dense spmm edge per second ",
          num_nodes * num_edge_per_node / output_time[1] / 1e9 * 1e3)


def test_dense_lstm():
    dev = torch.device("cuda:0")
    num_nodes = 10000
    num_edge_per_node = 32
    feat_len = 256
    lstm_module = torch.nn.LSTM(feat_len, feat_len, batch_first=True).cuda()
    a = torch.randn([num_nodes, num_edge_per_node, feat_len], device=dev)
    output_time = cxgc.prof("lstm", "dense", lambda: lstm_module(a))
    print("dense spmm edge per second ",
          num_nodes * num_edge_per_node / output_time[1] / 1e9 * 1e3)


def test_dense_mlp():
    dev = torch.device("cuda:0")
    num_nodes = 100000
    num_edge_per_node = 32
    feat_len = 256
    a = torch.randn([num_nodes, num_edge_per_node, feat_len], device=dev)
    weight = torch.randn([feat_len, feat_len], device=dev)
    output_time = cxgc.prof("mlp", "dense",
                            lambda: torch.sum(torch.matmul(a, weight), dim=1))
    print("dense mlp edge per second ",
          num_nodes * num_edge_per_node / output_time[1] / 1e9 * 1e3)


def test_edge_mlp():
    infeat = 256
    num_head = 1
    dset = "arxiv"
    dev = torch.device("cuda:0")
    feat, ptr, idx, b, edge_index = cxgc.prepare_data_full_graph(
        dset, feat_len=infeat, num_head=num_head, need_edge_index=True)
    num_type = 7
    weight = torch.randn([num_type, infeat, infeat], device=dev)
    output = torch.zeros_like(feat)
    types = torch.randint(0,
                          num_type, [edge_index.shape[1]],
                          device=dev,
                          dtype=torch.int32)
    output_time = cxgc.prof(
        "mlp", "edge", lambda: cxgnncomp_backend.typed_linear_s2d(
            feat, weight, output, edge_index[0], edge_index[1], types, 32))
    print("edge mlp edge per second ",
          edge_index.shape[1] / output_time[1] / 1e9 * 1e3)

    output_time = cxgc.prof(
        "mlp", "vertex", lambda: cxgnncomp_backend.aggr_rel_direct(
            feat, ptr, idx, weight, types, ptr.shape[0] - 1, num_type))
    print("vertex mlp edge per second ",
          edge_index.shape[1] / output_time[1] / 1e9 * 1e3)
    ptr, idx = cxgc.partition_2d(
        ptr,
        idx,
        b["num_node_in_layer"][-1],
        ptr.shape[0] - 1,
    )
    output_time = cxgc.prof(
        "mlp", "vertex", lambda: cxgnncomp_backend.aggr_rel_direct(
            feat, ptr, idx, weight, types, ptr.shape[0] - 1, num_type))
    print("2d mlp edge per second ",
          edge_index.shape[1] / output_time[1] / 1e9 * 1e3)


def test_add():
    infeat = 256
    num_head = 1
    dset = "arxiv"
    dev = torch.device("cuda:0")
    feat, ptr, idx, b, edge_index = cxgc.prepare_data_full_graph(
        dset, feat_len=infeat, num_head=num_head, need_edge_index=True)
    output = torch.empty([ptr.shape[0] - 1, infeat], device=dev)
    # edge impl
    output_time = cxgc.prof(
        "add", "edge", lambda: output.index_add_(
            0, edge_index[1], torch.index_select(feat, 0, edge_index[0])))
    print("add edge per second ",
          edge_index.shape[1] / output_time[1] / 1e9 * 1e3)
    # vertex impl
    tuner = cxgc.Tuner()
    tuner.tune_graph(ptr.shape[0] - 1, ptr.shape[0], feat.shape[-1],
                     cxgnncomp_backend.run_spmm_configurable,
                     [ptr, idx, feat, ptr.shape[0] - 1])
    output_time = cxgc.prof(
        "add", "vertex", lambda: tuner.tune_graph(
            ptr.shape[0] - 1, ptr.shape[0], feat.shape[-1], cxgnncomp_backend.
            run_spmm_configurable, [ptr, idx, feat, ptr.shape[0] - 1]))
    print("add vertex per second ",
          edge_index.shape[1] / output_time[1] / 1e9 * 1e3)

    ptr, idx = cxgc.partition_2d(
        ptr,
        idx,
        b["num_node_in_layer"][-1],
        ptr.shape[0] - 1,
    )
    tuner.tune_graph(ptr.shape[0] - 1, ptr.shape[0], feat.shape[-1],
                     cxgnncomp_backend.run_spmm_configurable,
                     [ptr, idx, feat, ptr.shape[0] - 1])
    output_time = cxgc.prof(
        "add", "2d", lambda: tuner.tune_graph(
            ptr.shape[0] - 1, ptr.shape[0], feat.shape[-1], cxgnncomp_backend.
            run_spmm_configurable, [ptr, idx, feat, ptr.shape[0] - 1]))
    print("add 2d per second ",
          edge_index.shape[1] / output_time[1] / 1e9 * 1e3)


def test_dense_overhead():
    # spmm (add)
    infeat = 256
    num_head = 1
    dset = "arxiv"
    dev = torch.device("cuda:0")
    feat, ptr, idx, b, edge_index = cxgc.prepare_data_full_graph(
        dset, feat_len=infeat, num_head=num_head, need_edge_index=True)
    output = torch.empty([ptr.shape[0] - 1, infeat], device=dev)
    val = torch.randn([edge_index.shape[1]], device=dev)
    for i in range(10):
        torch.cuda.synchronize()
        t0 = time.time()
        etensor = torch.index_select(feat, 0, idx)
        torch.cuda.synchronize()
        t1 = time.time()
        etensor = etensor * val.unsqueeze(-1)
        torch.cuda.synchronize()
        t2 = time.time()
        output.index_add_(0, edge_index[1], etensor)
        torch.cuda.synchronize()
        t3 = time.time()
        if i == 9:
            print("index select", t1 - t0, "mul", t2 - t1, "index add",
                  t3 - t2)
            print("ratio", ((t1 - t0) + (t3 - t2)) / (t3 - t0))
    # dot
    for i in range(10):
        torch.cuda.synchronize()
        t0 = time.time()
        etensor1 = torch.index_select(feat, 0, idx)
        etensor2 = torch.index_select(feat, 0, edge_index[1])
        torch.cuda.synchronize()
        t1 = time.time()
        torch.sum(etensor1 * etensor2, dim=1)
        torch.cuda.synchronize()
        t2 = time.time()
        if i == 9:
            print("index select", t1 - t0, "reduce", t2 - t1)
            print("ratio", ((t1 - t0)) / (t2 - t0))

    num_type = 7
    weight = torch.randn([num_type, infeat, infeat], device=dev)
    types = torch.randint(0,
                          num_type, [edge_index.shape[1]],
                          device=dev,
                          dtype=torch.int32)
    # mlp
    for i in range(10):
        torch.cuda.synchronize()
        t0 = time.time()
        etensor = torch.index_select(feat, 0, idx)
        # count = torch.bincount(types, minlength=num_type).cpu()
        arr = []
        for j in range(num_type):
            arr.append(etensor[types == j])

        torch.cuda.synchronize()
        t1 = time.time()
        for j in range(num_type):
            arr[j] = torch.matmul(arr[j], weight[j])
        torch.cuda.synchronize()
        t2 = time.time()
        for j in range(num_type):
            output.index_add_(0, edge_index[1][types == j], arr[j])
        torch.cuda.synchronize()
        t3 = time.time()
        if i == 9:
            print("index select", t1 - t0, "mm", t2 - t1, "index add", t3 - t2)
            print("ratio", ((t1 - t0) + (t3 - t2)) / (t3 - t0))


def process_edge_index(edge_index, block=256):
    src = edge_index[0]
    dst = edge_index[1]
    sorted_src, indices = torch.sort(src)
    sorted_dst = dst[indices]
    num_edge = edge_index.shape[1]
    num_src_node = torch.max(src).item() + 1
    num_dst_node = torch.max(sorted_dst).item() + 1
    src_deg = torch.bincount(src)
    new_src_deg = (src_deg + block - 1) // block * block
    new_num_edge = torch.sum(new_src_deg).item()
    src_deg = src_deg.cpu()
    new_src_deg = new_src_deg.cpu()
    print(f"padded edge num {new_num_edge}, original edge num {num_edge}")
    new_dst = torch.ones([new_num_edge], dtype=torch.int64) * -1
    new_src = torch.empty([new_num_edge], dtype=torch.int64)
    acc_padded = 0
    acc = 0
    edge_positions = torch.empty([num_edge], dtype=torch.int64)
    for i in range(num_src_node):
        new_src[acc_padded:acc_padded + new_src_deg[i]] = i
        new_dst[acc_padded:acc_padded + src_deg[i]] = sorted_dst[acc:acc +
                                                                 src_deg[i]]
        edge_positions[acc:acc + src_deg[i]] = torch.arange(
            acc_padded, acc_padded + src_deg[i])
        acc_padded += new_src_deg[i]
        acc += src_deg[i]
    assert acc == num_edge
    assert acc_padded == new_num_edge
    new_dst[new_dst == -1] = num_dst_node
    return torch.stack([new_src, new_dst]).to(
        edge_index.device), edge_positions.to(edge_index.device), torch.stack(
            [sorted_src, sorted_dst]).to(edge_index.device)


def test_process_edge_index():
    src = torch.tensor([3, 3, 1, 1, 1, 0, 0, 2, 2])
    dst = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
    edge_index = torch.stack([src, dst])
    output, edge_pos, sorted_output = process_edge_index(edge_index, block=4)
    print(output)
    print(sorted_output)


def show_padding_overhead(src):
    src_deg = torch.bincount(src)
    for i in range(32, 257, 32):
        new_src_deg = (src_deg + i - 1) // i * i
        new_num_edge = torch.sum(new_src_deg).item()
        print(
            f"block size {i}, padded edge num {new_num_edge}, original edge num {src.shape[0]}"
        )


def test_sddmm_triton():
    infeat = 256
    num_head = 16
    dev = torch.device("cuda:0")

    dset = "arxiv"
    feat, ptr, idx, b, edge_index = cxgc.prepare_data_full_graph(
        dset, feat_len=infeat, num_head=num_head, need_edge_index=True)
    transposed_feat = torch.transpose(feat, 1, 2).contiguous()
    transposed_feat = torch.cat([
        transposed_feat,
        torch.zeros([1, transposed_feat.shape[1], transposed_feat.shape[2]],
                    device=dev)
    ],
                                dim=0).contiguous()
    print(transposed_feat.shape)
    dst_feat = torch.randn([ptr.shape[0] - 1, infeat], device=feat.device)
    dst_feat = torch.cat(
        [dst_feat, torch.zeros([1, infeat], device=dev)], dim=0).contiguous()
    output = torch.zeros([idx.shape[0], num_head], device=feat.device)
    num_edge = edge_index.shape[1]

    # tmp_src = torch.arange(0, 1, device=dev)
    # tmp_src_repeated = torch.repeat_interleave(tmp_src, 1024)
    # tmp_src_repeated = torch.zeros([1024], device=dev, dtype=torch.int64)
    tmp_src_repeated = torch.repeat_interleave(
        torch.arange(0, 10, device=feat.device), 256)
    print(tmp_src_repeated)
    tmp_dst = edge_index[1][:tmp_src_repeated.shape[0]]
    tmp_output = torch.zeros([tmp_src_repeated.shape[0], num_head], device=dev)
    cxgnncomp_backend.run_sddmm(tmp_src_repeated, tmp_dst, feat, dst_feat,
                                tmp_output, tmp_src_repeated.shape[0])
    output_triton = cxgc.sddmm_dense(dst_feat, transposed_feat, tmp_dst,
                                     tmp_src_repeated, tmp_dst.shape[0])
    print(output_triton.shape)
    cxgc.compare(tmp_output, output_triton)
    # exit()

    show_padding_overhead(src=edge_index[0])

    new_edge_index, edge_pos, sorted_edge_index = process_edge_index(
        edge_index, block=32)

    cxgnncomp_backend.run_sddmm(sorted_edge_index[0], sorted_edge_index[1],
                                feat, dst_feat, output,
                                sorted_edge_index.shape[1])
    output2 = cxgc.sddmm_dense(dst_feat, transposed_feat, new_edge_index[1],
                               new_edge_index[0],
                               new_edge_index.shape[1])[edge_pos]
    cxgc.compare(output, output2)

    output_time = cxgc.prof(
        "sddmm", "edge parallel",
        lambda: cxgnncomp_backend.run_sddmm(edge_index[0], edge_index[
            1], feat, dst_feat, output, edge_index.shape[1]))

    cxgc.prof(
        "sddmm", "dense",
        lambda: cxgc.sddmm_dense(dst_feat, transposed_feat, new_edge_index[
            1], new_edge_index[0], new_edge_index.shape[1])[edge_pos])


def gen_data_neighbor():
    import os
    dset = "arxiv"
    feat, ptr, idx, b, edge_index = cxgc.prepare_data_full_graph(
        dset, feat_len=32, num_head=1, need_edge_index=True)
    new_ptr, new_target = cxgc.neighbor_grouping(ptr, 32)
    ptr = new_ptr
    new_edge_index = torch.stack([
        idx,
        torch.repeat_interleave(torch.arange(
            0, ptr.shape[0] - 1, device="cuda"),
                                repeats=ptr[1:] - ptr[:-1])
    ],
                                 dim=0)
    edge_index = edge_index.cpu().numpy()
    os.makedirs(f"/home/huangkz/data/dataset_diskgnn/{dset}-ng/processed",
                exist_ok=True)

    edge_index.tofile(
        f"/home/huangkz/data/dataset_diskgnn/{dset}-ng/processed/edge_index.dat"
    )
    ptr = ptr.cpu().numpy()
    ptr.tofile(
        f"/home/huangkz/data/dataset_diskgnn/{dset}-ng/processed/csr_ptr_undirected.dat"
    )
    idx = idx.cpu().numpy()
    idx.tofile(
        f"/home/huangkz/data/dataset_diskgnn/{dset}-ng/processed/csr_idx_undirected.dat"
    )


def gen_data():
    import os
    dset = "friendster"
    num_seed = 1000
    feat, ptr, idx, b, edge_index = cxgc.prepare_data_sampled_graph(
        dset=dset,
        feat_len=32,
        num_head=1,
        num_seeds=num_seed,
        need_edge_index=True)
    edge_index = edge_index.cpu().numpy()
    os.makedirs(
        f"/home/huangkz/data/dataset_diskgnn/{dset}-sample-{num_seed}/processed",
        exist_ok=True)

    edge_index.tofile(
        f"/home/huangkz/data/dataset_diskgnn/{dset}-sample-{num_seed}/processed/edge_index.dat"
    )
    ptr = ptr.cpu().numpy()
    ptr.tofile(
        f"/home/huangkz/data/dataset_diskgnn/{dset}-sample-{num_seed}/processed/csr_ptr_undirected.dat"
    )
    idx = idx.cpu().numpy()
    idx.tofile(
        f"/home/huangkz/data/dataset_diskgnn/{dset}-sample-{num_seed}/processed/csr_idx_undirected.dat"
    )


def gen_data_partition():
    import os
    dset = "papers100M"
    num_seed = 1000
    feat, ptr, idx, b, edge_index = cxgc.prepare_data_sampled_graph(
        dset=dset,
        feat_len=32,
        num_head=1,
        num_seeds=num_seed,
        need_edge_index=True)
    edge_index = edge_index.cpu().numpy()
    os.makedirs(
        f"/home/huangkz/data/dataset_diskgnn/{dset}-sample-{num_seed}/processed",
        exist_ok=True)

    edge_index.tofile(
        f"/home/huangkz/data/dataset_diskgnn/{dset}-sample-{num_seed}/processed/edge_index.dat"
    )
    ptr = ptr.cpu().numpy()
    ptr.tofile(
        f"/home/huangkz/data/dataset_diskgnn/{dset}-sample-{num_seed}/processed/csr_ptr_undirected.dat"
    )
    idx = idx.cpu().numpy()
    idx.tofile(
        f"/home/huangkz/data/dataset_diskgnn/{dset}-sample-{num_seed}/processed/csr_idx_undirected.dat"
    )


if __name__ == "__main__":
    # test_sddmm()
    # test_dense()
    # test_dense_spmm()
    # test_dense_lstm()
    # test_dense_mlp()
    # test_edge_mlp()
    # test_add()
    # test_dense_overhead()
    # test_process_edge_index()
    # test_sddmm_triton()

    gen_data()
    # gen_data_neighbor()
