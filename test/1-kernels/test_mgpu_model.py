import torch
import cxgnncomp as cxgc
import time
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import cxgnncomp_backend
import os
import logging

logger = logging.getLogger(__name__)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def check():
    assert graph.ptr.shape[
        0] - 1 <= num_local_dst_node, f"{rank} {i} {graph.ptr.shape[0] - 1} {num_local_dst_node}"
    assert torch.max(graph.idx) < feat.shape[
        0], f"{rank} {i} {feat.shape} {graph.idx.shape} {torch.max(graph.idx)} "
    assert graph.ptr[-1] == graph.idx.shape[
        0], f"{rank} {i} {graph.ptr[-1]} {graph.idx.shape[0]} "
    assert graph.ptr.device == feat.device
    assert graph.idx.device == feat.device
    assert graph.target.device == feat.device
    assert graph.target.shape[0] == graph.ptr.shape[0] - 1
    assert torch.all(graph.target >= 0)
    assert torch.max(
        graph.target
    ) < num_local_dst_node, f"{rank} {i} {torch.max(graph.target)} {num_local_dst_node} "


class P3DistAggrOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_emb, weight, graphs, num_src_node, num_dst_node,
                num_device, rank, layer):
        ctx.graphs = graphs
        ctx.num_device = num_device
        ctx.rank = rank
        ctx.num_src_node = num_src_node
        ctx.num_dst_node = num_dst_node
        graph = graphs[rank]
        # perform aggregate
        assert graph.ptr.shape[0] - 1 == num_dst_node
        assert torch.max(graph.idx) + 1 < num_src_node
        output = torch.zeros(num_dst_node,
                             input_emb.shape[1],
                             device=input_emb.device)
        cxgnncomp_backend.target_aggr(input_emb, graph.ptr.to(torch.int64),
                                      graph.idx.to(torch.int64),
                                      graph.target.to(torch.int64), output,
                                      graph.ptr.shape[0] - 1)

        extra = num_dst_node % num_device
        ctx.save_for_backward(output)
        output = torch.mm(output, weight)
        # dist.all_reduce(output)
        my_num_node = num_dst_node // num_device + (extra if rank
                                                    == num_device - 1 else 0)
        output_tensor = torch.empty([my_num_node, output.shape[1]],
                                    device=output.device)
        if extra == 0:
            dist.reduce_scatter_tensor(output_tensor, output)
        else:
            dist.reduce(output[-extra:], num_device - 1)
            if rank == num_device - 1:
                dist.reduce_scatter(output_tensor[:-extra], output[:-extra])
                output_tensor[-extra:] = output[-extra:]
            else:
                dist.reduce_scatter(output_tensor, output[:-extra])
        return output_tensor

    @staticmethod
    def backward(ctx, grad_in):
        graphs = ctx.graphs
        num_device = ctx.num_device
        rank = ctx.rank
        num_src_node = ctx.num_src_node
        num_dst_node = ctx.num_dst_node
        extra = num_dst_node % num_device
        recv_buf = torch.zeros([num_dst_node, grad_in.shape[1]],
                               device=grad_in.device)
        if extra == 0:
            dist.all_gather(recv_buf, grad_in)
        else:
            if rank == num_device - 1:
                dist.all_gather(recv_buf[:-extra], grad_in[:-extra])
                recv_buf[-extra:] = grad_in[-extra:]
                dist.broadcast(recv_buf[-extra:], num_device - 1)
            else:
                dist.all_gather(recv_buf[:-extra], grad_in)
                dist.broadcast(recv_buf[-extra:], num_device - 1)
        grad = torch.mm(torch.t(ctx.saved_tensors[0]), recv_buf)
        return grad, None, None, None, None, None, None


class DistAggrOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_emb, graphs, num_src_node, num_dst_node, num_device,
                rank, layer):
        ctx.graphs = graphs
        ctx.num_device = num_device
        ctx.rank = rank
        ctx.num_local_src_node = input_emb.shape[0]
        ctx.layer = layer
        num_local_dst_node = num_dst_node // num_device if rank != num_device - 1 else num_dst_node // num_device + (
            num_dst_node % num_device)
        output = torch.zeros(num_local_dst_node,
                             input_emb.shape[1],
                             device=rank)
        num_max_remote = num_src_node // num_device + (num_src_node %
                                                       num_device)
        if layer > 0:
            remote_feat = torch.empty([num_max_remote, input_emb.shape[1]],
                                      device=rank)
        print(
            f"rank {rank} forward {layer} {input_emb.shape} {output.shape} {torch.cuda.memory_allocated(rank)}"
        )
        for i in range(num_device):
            graph = graphs[i]
            if i == rank:
                if layer == 0:
                    feat = input_emb.to(rank)
                else:
                    feat = input_emb
            else:
                num_remote_src_node = num_src_node // num_device if i != num_device - 1 else num_src_node // num_device + (
                    num_src_node % num_device)
                if layer == 0:
                    feat = torch.empty(
                        size=[num_remote_src_node, input_emb.shape[1]],
                        device=rank)
                else:
                    feat = remote_feat[:num_remote_src_node]
            dist.broadcast(feat, i)

            cxgnncomp_backend.target_aggr(
                feat,
                graph.ptr.to(rank).to(torch.int64),
                graph.idx.to(rank).to(torch.int64),
                graph.target.to(rank).to(torch.int64), output,
                graph.ptr.shape[0] - 1)
        return output

    @staticmethod
    def backward(ctx, grad_in):
        graphs = ctx.graphs
        num_device = ctx.num_device
        rank = ctx.rank
        layer = ctx.layer
        print(
            f"backward rank {rank} layer {layer} {torch.cuda.memory_allocated(rank)}"
        )
        if layer == 0:
            return None, None, None, None, None, None, None
        num_local_src_node = ctx.num_local_src_node
        final_grad = None
        grad_out = torch.empty([num_local_src_node, grad_in.shape[1]],
                                device=grad_in.device)
        for i in range(num_device):
            grad_out.zero_()
            graph = graphs[i]
            cxgnncomp_backend.target_aggr_backward(
                grad_in,
                graph.ptr.to(rank).to(torch.int64),
                graph.idx.to(rank).to(torch.int64),
                graph.target.to(rank).to(torch.int64), grad_out,
                graph.ptr.shape[0] - 1)
            dist.reduce(grad_out, i)
            if i == rank:
                final_grad = grad_out.detach().clone()

        return final_grad, None, None, None, None, None, None


class DistAggrOPDensify(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_emb, graphs, num_src_node, num_dst_node, num_device,
                rank, idx_needed_layered):
        ctx.graphs = graphs
        ctx.num_device = num_device
        ctx.rank = rank
        ctx.num_local_src_node = input_emb.shape[0]
        ctx.idx_needed_layered = idx_needed_layered
        num_local_dst_node = num_dst_node // num_device if rank != num_device - 1 else num_dst_node // num_device + (
            num_dst_node % num_device)
        output = torch.zeros(num_local_dst_node,
                             input_emb.shape[1],
                             device=input_emb.device)
        # local computation
        logger.info("begin local computation")
        cxgnncomp_backend.target_aggr(input_emb,
                                      graphs[rank].ptr.to(torch.int64),
                                      graphs[rank].idx.to(torch.int64),
                                      graphs[rank].target.to(torch.int64),
                                      output, graphs[rank].ptr.shape[0] - 1)
        logger.info("end local computation")
        torch.cuda.synchronize(rank)
        for receiver in range(num_device):
            for sender in range(num_device):
                if receiver == rank:
                    if sender == rank:
                        continue
                    else:
                        torch.cuda.synchronize(rank)
                        graph = graphs[sender]
                        buf = torch.empty(
                            [graph.num_unique, input_emb.shape[1]],
                            device=input_emb.device)
                        dist.recv(buf, sender)
                        cxgnncomp_backend.target_aggr(
                            buf, graph.ptr.to(torch.int64),
                            graph.idx.to(torch.int64),
                            graph.target.to(torch.int64), output,
                            graph.ptr.shape[0] - 1)
                        torch.cuda.synchronize(rank)
                        del buf
                if sender == rank:
                    if receiver == rank:
                        continue
                    else:
                        # print(
                        #     f"rank {rank} send to {receiver} {torch.cuda.memory_allocated(rank)} {ctx.idx_needed_layered[receiver].shape} {input_emb.shape}",
                        #     flush=True)
                        torch.cuda.synchronize(rank)
                        buf = input_emb[ctx.idx_needed_layered[receiver]]
                        torch.cuda.synchronize(rank)
                        dist.send(buf, receiver)
                        torch.cuda.synchronize(rank)
                        del buf

        return output

    @staticmethod
    def backward(ctx, grad_in):
        graphs = ctx.graphs
        num_device = ctx.num_device
        rank = ctx.rank
        num_local_src_node = ctx.num_local_src_node
        idx_needed_layered = ctx.idx_needed_layered
        grad_out = torch.zeros([num_local_src_node, grad_in.shape[1]],
                               device=grad_in.device)
        for receiver in range(num_device):
            for sender in range(num_device):
                if receiver == rank and sender == rank:  # local computation
                    graph = graphs[rank]
                    cxgnncomp_backend.target_aggr_backward(
                        grad_in, graph.ptr.to(torch.int64),
                        graph.idx.to(torch.int64),
                        graph.target.to(torch.int64), grad_out,
                        graph.ptr.shape[0] - 1)
                elif receiver == rank:
                    graph = graphs[sender]
                    buf = torch.empty([
                        idx_needed_layered[sender].shape[0], grad_in.shape[1]
                    ],
                                      device=grad_in.device)
                    dist.recv(buf, sender)
                    torch.cuda.synchronize(rank)
                    grad_out.index_add_(0, idx_needed_layered[sender], buf)
                    torch.cuda.synchronize(rank)
                    del buf
                elif sender == rank:
                    graph = graphs[receiver]
                    buf = torch.zeros([graph.num_unique, grad_in.shape[1]],
                                      device=grad_in.device)
                    cxgnncomp_backend.target_aggr_backward(
                        grad_in, graph.ptr.to(torch.int64),
                        graph.idx.to(torch.int64),
                        graph.target.to(torch.int64), buf,
                        graph.ptr.shape[0] - 1)
                    torch.cuda.synchronize(rank)
                    dist.send(buf, receiver)
                    torch.cuda.synchronize(rank)
                    del buf

        return grad_out, None, None, None, None, None, None


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
            torch.nn.Linear(in_channels, hidden_channels,
                            bias=False).cuda(rank))
        self.channels = [self.in_channels]
        for i in range(num_layer - 2):
            self.linears.append(
                torch.nn.Linear(hidden_channels, hidden_channels,
                                bias=False).cuda(rank))
            self.channels.append(hidden_channels)
        self.linears.append(
            torch.nn.Linear(hidden_channels, out_channels,
                            bias=False).cuda(rank))
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

    def forward_cxg(self, graphs, rank, local_feat, num_node_in_layer,
                    idx_needed_layered, config):
        # init layer
        input_emb = local_feat
        for l in range(self.num_layer):
            torch.cuda.synchronize(rank)
            # torch.cuda.empty_cache()
            print(f"rank {rank} layer {l} {input_emb.shape}",
                  torch.cuda.memory_allocated(rank))
            dist.barrier()
            if config["skip_input"] and l == 0:
                output = input_emb
            else:
                if config["densify"]:
                    output = DistAggrOPDensify.apply(
                        input_emb, graphs[l], num_node_in_layer[-1 - l],
                        num_node_in_layer[-1 - l - 1], self.num_device, rank,
                        idx_needed_layered[l])
                else:
                    output = DistAggrOP.apply(input_emb, graphs[l],
                                              num_node_in_layer[-1 - l],
                                              num_node_in_layer[-1 - l - 1],
                                              self.num_device, rank, l)
            input_emb = self.linears[l](output)
        return input_emb

    def forward(self, graphs, rank, local_feat, num_node_in_layer,
                idx_needed_layered, config):
        if self.graph_type.lower() == "csr_layer":
            return self.forward_cxg(graphs, rank, local_feat,
                                    num_node_in_layer, idx_needed_layered,
                                    config)
        else:
            print(self.graph_type.lower())
            raise NotImplementedError


def train(graphs, rank, local_feat, model, optimizer, lossfn,
          num_node_in_layer, idx_needed_layered, config):
    t0 = time.time()
    optimizer.zero_grad()
    # forward
    output = model(graphs, rank, local_feat, num_node_in_layer,
                   idx_needed_layered, config)
    # loss
    loss = lossfn(output, torch.randn_like(output))
    # backward
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(rank)
    # if rank == 0:
    print("time elapsed: ", rank, time.time() - t0)


def process_idx(idx, get_unique=False):
    unique_idx = torch.unique(idx)
    sorted_unique_idx, indices = torch.sort(unique_idx)
    idx = torch.searchsorted(sorted_unique_idx, idx)
    if not get_unique:
        return idx, unique_idx.shape[0]
    else:
        return sorted_unique_idx


def run(rank, world_size, args):

    if rank == 0:
        logger.setLevel(logging.INFO if args.log_level ==
                        "INFO" else logging.WARN)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.setLevel(logging.WARN)
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
        # print("bincount", torch.bincount(batch["visit_mask"] + 1))

        # reorder graph for locality
        if args.reorder_file != "":
            new_order = torch.from_numpy(
                np.fromfile(args.reorder_file + "reorder_" + args.dataset +
                            ".dat",
                            dtype=np.int64)).to(edge_index.device)
            edge_index = new_order[edge_index]
            new_mask = torch.ones_like(batch["visit_mask"]) * -1
            for i in range(0, args.num_layer + 1):
                new_mask[new_order[batch["visit_mask"] == i]] = i
            batch["visit_mask"] = new_mask
            # print("bincount", torch.bincount(batch["visit_mask"] + 1))
            # arr = torch.arange(new_order.shape[0], device=new_order.device)
            # print(batch["visit_mask"].dtype)
            # arr = arr[batch["visit_mask"]]
            # arr = new_order[arr]
            # batch["visit_mask"] = torch.zeros_like(batch["visit_mask"])
            # batch["visit_mask"] = new_order[batch["visit_mask"]]

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
            if args.skip_input and l == 0:
                continue
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
        if args.skip_input and l == 0:
            continue
        for it, item in enumerate(graphs_in_layer[l]):
            item.to(rank)
            item.ptr = item.ptr.to(torch.int32)
            item.idx = item.idx.to(torch.int32)
            item.target = item.target.to(torch.int32)
            if it != rank:  # densify for remote graph partition
                item.idx, item.num_unique = process_idx(item.idx)
            # print(
            #     f"rank {rank} layer {l} part {it} {torch.unique(item.idx).shape[0]} {num_node_in_layer[-1 - l] / args.num_device}"
            # )

    idx_needed_layered = [[] for _ in range(args.num_layer)]
    if args.densify:
        for l in range(args.num_layer):
            if args.skip_input and l == 0:
                continue
            for j in range(args.num_device):
                if j == rank:
                    idx_needed_layered[l].append(None)
                    continue
                b = cxgc.Batch(x=None, ptr=None, idx=None, target=None)
                b.fromfile(
                    f".cache/{args.dataset}_{args.num_device}_{j}_{l}_{rank}")
                idx_needed_layered[l].append(
                    process_idx(b.idx.to(rank), get_unique=True))

    cpu_graph_data = True
    if cpu_graph_data:
        for l in range(args.num_layer):
            for it, item in enumerate(graphs_in_layer[l]):
                item.to("cpu")

    # if rank == 0:
    print(rank, "prepare graph dataset", torch.cuda.memory_allocated(rank))
    # prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    lossfn = torch.nn.MSELoss()

    # prepare local features
    if args.skip_input:
        num_total_node = num_node_in_layer[-2]
    else:
        num_total_node = num_node_in_layer[-1]
    if args.baseline.lower() == "p3":
        if rank != world_size - 1:
            local_feat = torch.randn(
                [num_total_node, args.infeat // world_size],
                device=rank,
                requires_grad=False)
        else:
            local_feat = torch.randn([
                num_total_node, args.infeat // world_size +
                (args.infeat % world_size)
            ],
                                     device=rank,
                                     requires_grad=False)
    else:
        if rank != world_size - 1:
            local_feat = torch.randn(
                [num_total_node // world_size, args.infeat],
                device="cpu",
                requires_grad=False)
        else:
            local_feat = torch.randn([
                num_total_node // world_size +
                (num_total_node % world_size), args.infeat
            ],
                                     device="cpu",
                                     requires_grad=False)

    # if rank == 0:
    print(rank, "prepare local feat", torch.cuda.memory_allocated(rank))

    config = {
        "skip_input": args.skip_input,
        "densify": args.densify,
    }
    # train
    for i in range(args.num_epoch):
        train(graphs_in_layer, rank, local_feat, model, optimizer, lossfn,
              num_node_in_layer, idx_needed_layered, config)
        torch.cuda.empty_cache()

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
    parser.add_argument("--skip_input", type=int, default=0)
    parser.add_argument("--densify", type=int, default=1)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--baseline", type=str, default="")
    args = parser.parse_args()
    print(args)
    mp.spawn(run,
             args=(args.num_device, args),
             nprocs=args.num_device,
             join=True)
