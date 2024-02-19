import torch
import cxgnncomp as cxgc
import cxgnncomp_backend


def run(head, feat, hidden, edge, node, rel, node_dst):
    # RGCN
    nn_workload = feat * hidden * edge
    graph_workload = (hidden + feat) * edge

    nn_workload_opt = node * feat * hidden
    graph_workload_opt = node * feat + edge * hidden
    print(
        f"RGCN\t{nn_workload}\t{graph_workload}\t{nn_workload_opt}\t{graph_workload_opt}"
    )

    # GAT
    nn_workload = feat * hidden * node * head + node * head * hidden
    graph_workload = edge * head * hidden * 2

    nn_workload_opt = (
        feat * hidden * head + feat * node * head + (feat * hidden * node * head)
    )
    graph_workload_opt = edge * head * feat * 2

    print(
        f"GAT\t{nn_workload}\t{graph_workload}\t{nn_workload_opt}\t{graph_workload_opt}"
    )

    # SAGE
    nn_workload = feat * hidden * node + feat * hidden * node_dst
    graph_workload = feat * edge * 2

    nn_workload_opt = node_dst * feat * hidden * 2
    graph_workload_opt = min(feat, hidden) * edge * 2

    print(
        f"SAGE\t{nn_workload}\t{graph_workload}\t{nn_workload_opt}\t{graph_workload_opt}"
    )


def exec(x, ptr, idx, feat, hidden, head):
    batch = torch.load("../../papers100M-graphsaint.pt")
    ptr = batch["ptr"].cuda()
    idx = batch["idx"].cuda()
    num_node_in_layer = batch["num_node_in_layer"]
    x = torch.randn([num_node_in_layer[-1], feat], device=x.device)
    weight = torch.randn([feat, hidden], device=x.device)
    cxgc.prof("mm", "node", lambda: torch.mm(x, weight))
    cxgc.prof("mm", "node", lambda: torch.mm(x, weight))
    res_nn = cxgc.prof("mm", "node", lambda: torch.mm(x, weight))
    res_graph = cxgc.prof(
        "spmm",
        "",
        lambda: cxgnncomp_backend.sage_mean_forward(x, ptr, idx, num_node_in_layer[-2]),
    )
    print(f"nn: {res_nn[0]}, graph: {res_graph[0]}")
    print(
        f"nn: {res_nn[0]/(hidden * feat * x.shape[0])}, graph: {res_graph[0] /(feat * idx.shape[0])}"
    )
    # tuner = cxgc.Tuner()
    # output = tuner.tune_graph(ptr.shape[0] - 1, idx.shape[0], x.shape[1],
    #                           cxgnncomp_backend.run_spmm_configurable,
    #                           [ptr, idx, x, ptr.shape[0] - 1])


# data prepare
head = 16
feat = 128
hidden = 256
edge = 2332486
node = 169343
rel = 7
run(head, feat, hidden, edge, node, rel, node_dst=node)

dset = "papers100M"
x, ptr, idx, b = cxgc.prepare_data_sampled_graph(
    dset=dset, feat_len=128, num_head=1, num_seeds=1000, need_edge_index=False
)
feat = 128
node = b["num_node_in_layer"][-1]
node_dst = b["num_node_in_layer"][-2]
edge = idx.shape[0]
run(head, feat, hidden, edge, node, rel, node_dst=node_dst)
exec(x, ptr, idx, feat, hidden, head)
