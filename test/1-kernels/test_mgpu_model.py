import torch
import cxgnncomp as cxgc
import time
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import cxgnncomp_backend
import os


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def check():
    assert torch.max(graph.idx) < feat.shape[
        0], f"{rank} {i} {feat.shape} {graph.idx.shape} {torch.max(graph.idx)} {num_total_node}"
    assert graph.ptr[-1] == graph.idx.shape[
        0], f"{rank} {i} {graph.ptr[-1]} {graph.idx.shape[0]} {num_total_node}"
    assert graph.ptr.device == local_feat.device
    assert graph.idx.device == local_feat.device
    assert graph.target.device == local_feat.device
    assert graph.target.shape[0] == graph.ptr.shape[0] - 1
    assert torch.max(graph.target) < local_feat.shape[
        0], f"{rank} {i} {torch.max(graph.target)} {local_feat.shape[0]} {num_total_node}"


class Model(cxgc.GNN):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layer,
                 dropout, graph_type, num_device, rank, **kwargs):
        super().__init__(in_channels,
                         hidden_channels,
                         out_channels,
                         num_layer,
                         dropout,
                         graph_type,
                         config=None,
                         **kwargs)
        self.num_device = num_device
        self.linears = []
        self.linears.append(
            torch.nn.Linear(in_channels, hidden_channels).cuda(rank))
        self.channels = [self.in_channels]
        for i in range(num_layer - 2):
            self.linears.append(
                torch.nn.Linear(hidden_channels, hidden_channels).cuda(rank))
            self.channels.append(hidden_channels)
        self.linears.append(
            torch.nn.Linear(hidden_channels, out_channels).cuda(rank))
        self.channels.append(hidden_channels)

    def reset_parameters(self):
        for linear in self.linears:
            linear.reset_parameters()

    def forward_roc(self, batch):
        x = batch.x
        for i in range(self.num_layer):
            x = self.convs[i](batch, x)
        return x

    def forward_dgl(self, blocks, x):
        for i in range(self.num_layer):
            x = self.convs[i](blocks[i], x)
        return x

    def forward_cxg(self, graphs, rank, local_feat, num_total_node):
        # init layer
        input_emb = local_feat
        local_num_node = input_emb.shape[0]
        device = input_emb.device
        for l in range(self.num_layer):
            output = torch.zeros([local_num_node, input_emb.shape[1]],
                                 device=device)
            for i in range(self.num_device):
                graph = graphs[i]
                if i == rank:
                    feat = input_emb
                else:
                    num_remote_node = num_total_node // self.num_device if i != self.num_device - 1 else num_total_node // self.num_device + (
                        num_total_node % self.num_device)
                    feat = torch.empty(
                        size=[num_remote_node, self.channels[l]],
                        device=local_feat.device)
                dist.broadcast(feat, i)
                cxgnncomp_backend.target_aggr(feat, graph.ptr, graph.idx,
                                              graph.target, output,
                                              graph.ptr.shape[0] - 1)
            del feat
            # torch.cuda.synchronize(rank)
            # print(
            #     f"rank {rank} layer {l} output {output.shape} num_layer {self.num_layer}"
            # )
            input_emb = self.linears[l](output)

    def forward(self, graphs, rank, local_feat, num_total_node):
        if self.graph_type.lower() == "csr_layer":
            self.forward_cxg(graphs, rank, local_feat, num_total_node)
        else:
            print(self.graph_type.lower())
            raise NotImplementedError


def train(graphs, rank, local_feat, model, optimizer, lossfn, num_total_node):
    t0 = time.time()
    optimizer.zero_grad()
    # forward
    output = model(graphs, rank, local_feat, num_total_node)
    # loss
    # loss = lossfn(output, torch.randn_like(output))
    # backward
    # loss.backward()
    # optimizer.step()
    torch.cuda.synchronize(rank)
    # if rank == 0:
    print("time elapsed: ", rank, time.time() - t0)


def run(rank, world_size, args):
    setup(rank, world_size)

    feat, ptr, idx, b, edge_index = cxgc.prepare_graph(
        dset=args.dataset,
        feat_len=args.infeat,
        num_head=args.num_head,
        num_seeds=args.num_seeds,
        need_edge_index=True,
        is_full_graph=args.is_full_graph,
        device="cpu",
        need_feat=False)
    num_node = ptr.shape[0] - 1

    model = Model(
        args.infeat,
        args.hiddenfeat,
        args.outfeat,
        graph_Type=args.graph_type,
        num_layer=args.num_layer,
        dropout=0.5,
        graph_type=args.graph_type,
        num_device=args.num_device,
        rank=rank,
    )
    model.reset_parameters()

    # reorder graph for locality
    if args.reorder_file != "":
        new_order = torch.from_numpy(
            np.fromfile(args.reorder_file + "reorder_" + args.dataset +
                        ".dat")).to(rank)
        # reorder ptr and idx
        raise NotImplementedError

    # prepare local features
    if rank != world_size - 1:
        local_feat = torch.randn([num_node // world_size,
                                  args.infeat]).to(rank)
    else:
        local_feat = torch.randn(
            [num_node // world_size + (num_node % world_size),
             args.infeat]).to(rank)

    # prepare graphs
    graphs = []
    for remote_rank in range(args.num_device):
        ptr, idx, target = cxgc.partition_2d_gpu(edge_index,
                                                 args.num_device,
                                                 rank,
                                                 remote_rank,
                                                 device=rank)
        graphs.append(cxgc.Batch(x=None, ptr=ptr, idx=idx, target=target))
    del edge_index
    for item in graphs:
        item.to(rank)

    # prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    lossfn = torch.nn.MSELoss()

    # train
    for i in range(args.num_epoch):
        train(graphs, rank, local_feat, model, optimizer, lossfn, num_node)

    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--graph_type", type=str, default="CSR_Layer")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--hiddenfeat", type=int, default=256)
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--num_head", type=int, default=1)
    parser.add_argument("--num_rel", type=int, default=7)
    parser.add_argument("--infeat", type=int, default=64)
    parser.add_argument("--outfeat", type=int, default=64)
    parser.add_argument("--is_full_graph", type=int, default=1)
    parser.add_argument("--num_seeds", type=int, default=1000)
    parser.add_argument("--reorder_file",
                        type=str,
                        default="/home/huangkz/repos/rabbit_order/demo/")
    parser.add_argument("--num_device", type=int, default=4)
    parser.add_argument("--num_epoch", type=int, default=10)
    args = parser.parse_args()
    print(args)
    mp.spawn(run,
             args=(args.num_device, args),
             nprocs=args.num_device,
             join=True)
