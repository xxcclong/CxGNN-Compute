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


class DistAggrOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_emb, graphs, num_src_node, num_dst_node, num_device,
                rank):
        ctx.graphs = graphs
        ctx.num_device = num_device
        ctx.rank = rank
        ctx.num_local_src_node = input_emb.shape[0]
        # ctx.save_for_backward(num_total_node, num_device, rank)
        num_local_dst_node = num_dst_node // num_device if rank != num_device - 1 else num_dst_node // num_device + (
            num_dst_node % num_device)
        output = torch.zeros(num_local_dst_node,
                             input_emb.shape[1],
                             device=input_emb.device)
        for i in range(num_device):
            graph = graphs[i]
            if i == rank:
                feat = input_emb
            else:
                num_remote_src_node = num_src_node // num_device if i != num_device - 1 else num_src_node // num_device + (
                    num_src_node % num_device)
                feat = torch.empty(
                    size=[num_remote_src_node, input_emb.shape[1]],
                    device=input_emb.device)
            dist.broadcast(feat, i)

            # assert graph.ptr.shape[
            #     0] - 1 <= num_local_dst_node, f"{rank} {i} {graph.ptr.shape[0] - 1} {num_local_dst_node}"
            # assert torch.max(graph.idx) < feat.shape[
            #     0], f"{rank} {i} {feat.shape} {graph.idx.shape} {torch.max(graph.idx)} "
            # assert graph.ptr[-1] == graph.idx.shape[
            #     0], f"{rank} {i} {graph.ptr[-1]} {graph.idx.shape[0]} "
            # assert graph.ptr.device == feat.device
            # assert graph.idx.device == feat.device
            # assert graph.target.device == feat.device
            # assert graph.target.shape[0] == graph.ptr.shape[0] - 1
            # assert torch.all(graph.target >= 0)
            # assert torch.max(
            #     graph.target
            # ) < num_local_dst_node, f"{rank} {i} {torch.max(graph.target)} {num_local_dst_node} "

            cxgnncomp_backend.target_aggr(feat, graph.ptr.to(torch.int64),
                                          graph.idx.to(torch.int64),
                                          graph.target.to(torch.int64), output,
                                          graph.ptr.shape[0] - 1)
            torch.cuda.synchronize(rank)
        return output

    @staticmethod
    def backward(ctx, grad_in):
        graphs = ctx.graphs
        num_device = ctx.num_device
        rank = ctx.rank
        num_local_src_node = ctx.num_local_src_node
        final_grad = None
        for i in range(num_device):
            grad_out = torch.zeros([num_local_src_node, grad_in.shape[1]],
                                   device=grad_in.device)
            graph = graphs[i]
            cxgnncomp_backend.target_aggr_backward(
                grad_in, graph.ptr.to(torch.int64), graph.idx.to(torch.int64),
                graph.target.to(torch.int64), grad_out, graph.ptr.shape[0] - 1)
            dist.reduce(grad_out, i)
            if i == rank:
                final_grad = grad_out

        return final_grad, None, None, None, None, None


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

    def forward_cxg(self, graphs, rank, local_feat, num_node_in_layer):
        # init layer
        input_emb = local_feat
        for l in range(self.num_layer):
            # if rank == 0:
            torch.cuda.synchronize(rank)
            # torch.cuda.empty_cache()
            print(f"rank {rank} layer {l} {input_emb.shape}",
                  torch.cuda.memory_allocated(rank))
            dist.barrier()
            output = DistAggrOP.apply(input_emb, graphs[l],
                                      num_node_in_layer[-1 - l],
                                      num_node_in_layer[-1 - l - 1],
                                      self.num_device, rank)
            print(f"rank {rank} layer {l} {output.shape} {self.linears[l]}")
            input_emb = self.linears[l](output)
        return input_emb

    def forward(self, graphs, rank, local_feat, num_node_in_layer):
        if self.graph_type.lower() == "csr_layer":
            return self.forward_cxg(graphs, rank, local_feat,
                                    num_node_in_layer)
        else:
            print(self.graph_type.lower())
            raise NotImplementedError


def train(graphs, rank, local_feat, model, optimizer, lossfn,
          num_node_in_layer):
    t0 = time.time()
    optimizer.zero_grad()
    # forward
    output = model(graphs, rank, local_feat, num_node_in_layer)
    # loss
    loss = lossfn(output, torch.randn_like(output))
    # backward
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(rank)
    # if rank == 0:
    print("time elapsed: ", rank, time.time() - t0)


def run(rank, world_size, args):
    setup(rank, world_size)

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

    # prepare graphs
    '''
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
    '''
    if args.gen_cache:
        feat, ptr, idx, batch, edge_index = cxgc.prepare_graph(
            dset=args.dataset,
            feat_len=args.infeat,
            num_head=args.num_head,
            num_seeds=args.num_seeds,
            need_edge_index=True,
            is_full_graph=args.is_full_graph,
            device="cpu",
            need_feat=False,
            rank=rank)
        del ptr, idx
        graphs_in_layer = [[] for _ in range(args.num_layer)]
        visit_mask = batch["visit_mask"]  # .to(rank)
        print(edge_index.shape)
        # edge_index = edge_index.to(rank)
        for l in range(args.num_layer):
            for i in range(args.num_device):
                ptr, idx, target = cxgc.partition_2d_gpu_layered(
                    edge_index,
                    args.num_device,
                    rank_dst=rank,
                    rank_src=i,
                    visit_mask=visit_mask,
                    layer_id=l)
                graphs_in_layer[l].append(
                    cxgc.Batch(x=None, ptr=ptr, idx=idx, target=target))
        import os
        for l in range(args.num_layer):
            for it, item in enumerate(graphs_in_layer[l]):
                # create directory with dataset name
                dir = f".cache/{args.dataset}_{args.num_device}_{rank}_{l}_{it}"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                item.tofile(dir)
        with open(f".cache/{args.dataset}_num_node_in_layer.txt", 'w') as f:
            for item in batch["num_node_in_layer"]:
                print(item, file=f)
        return
    else:
        graphs_in_layer = [[] for _ in range(args.num_layer)]
        for l in range(args.num_layer):
            for i in range(args.num_device):
                b = cxgc.Batch(x=None, ptr=None, idx=None, target=None)
                b.fromfile(
                    f".cache/{args.dataset}_{args.num_device}_{rank}_{l}_{i}")
                graphs_in_layer[l].append(b)
        with open(f".cache/{args.dataset}_num_node_in_layer.txt", 'r') as f:
            num_node_in_layer = [int(x) for x in f.readlines()]
            if rank == 0:
                print("num_node_in_layer", num_node_in_layer)

    for l in range(args.num_layer):
        for item in graphs_in_layer[l]:
            item.to(rank)
            item.ptr = item.ptr.to(torch.int32)
            item.idx = item.idx.to(torch.int32)
            item.target = item.target.to(torch.int32)
            # print(l, item.ptr.shape[0] - 1, num_node_in_layer)
    # if rank == 0:
    print(rank, "prepare graph dataset", torch.cuda.memory_allocated(rank))
    # prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    lossfn = torch.nn.MSELoss()

    # prepare local features
    if rank != world_size - 1:
        local_feat = torch.randn(
            [num_node_in_layer[-1] // world_size, args.infeat]).to(rank)
    else:
        local_feat = torch.randn([
            num_node_in_layer[-1] // world_size +
            (num_node_in_layer[-1] % world_size), args.infeat
        ]).to(rank)

    # if rank == 0:
    print(rank, "prepare local feat", torch.cuda.memory_allocated(rank))

    # train
    for i in range(args.num_epoch):
        train(graphs_in_layer, rank, local_feat, model, optimizer, lossfn,
              num_node_in_layer)

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
    parser.add_argument("--gen_cache", type=int, default=0)
    args = parser.parse_args()
    print(args)
    mp.spawn(run,
             args=(args.num_device, args),
             nprocs=args.num_device,
             join=True)
