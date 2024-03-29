import torch
import cxgnncomp as cxgc
import time
import torch_geometric.nn as pygnn

in_feat = 32


def prepare_data():
    dset = "arxiv"
    num_head = 1
    x, ptr, idx, b, edge_index = cxgc.prepare_data_full_graph(
        dset,
        feat_len=in_feat,
        num_head=num_head,
        need_edge_index=True,
    )
    # x, ptr, idx, b = cxgc.prepare_data_sampled_graph(dset=dset,
    #                                                  feat_len=in_feat,
    #                                                  num_head=num_head,
    #                                                  num_seeds=1000)
    return x, ptr, idx, b, num_head, edge_index


def lstm_pyg():
    conv = pygnn.SAGEConv(in_feat, in_feat, aggr="lstm").cuda()
    cxgc.prof("lstm", "pyg", lambda: conv(x, edge_index))
    exit()


x, ptr, idx, batch, num_head, edge_index = prepare_data()

# lstm_pyg()
num_edge = idx.shape[0]
num_center = ptr.shape[0] - 1
deg = ptr[1:] - ptr[:-1]
count = torch.bincount(deg).cpu()
print(
    "num_edge",
    num_edge,
    "num_center",
    num_center,
    "deg",
    deg.shape,
)

lstm_module = torch.nn.LSTM(in_feat, in_feat, batch_first=True).cuda()

import dgl.function as fn
import dgl


def lstm_dgl():

    def _lstm_reducer(nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        # h = (
        #     m.new_zeros((1, batch_size, self._in_src_feats)),
        #     m.new_zeros((1, batch_size, self._in_src_feats)),
        # )
        _, (rst, _) = lstm_module(
            m,
        )
        return {"neigh": rst.squeeze(0)}

    num_src = batch["num_node_in_layer"][-1]
    num_dst = batch["num_node_in_layer"][-2]
    # dgl_graph = dgl.create_block(('csc', (ptr, idx, torch.tensor([]))),
    #                              int(num_src), int(num_dst))
    # print(dgl_graph.ntypes)
    dgl_graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_src).to("cuda")
    dgl_graph.ndata["h"] = x
    msg_fn = fn.copy_src("h", "m")
    # print(dgl_graph.ntypes)
    dgl_graph.update_all(msg_fn, _lstm_reducer)

    cxgc.prof("lstm", "dgl", lambda: dgl_graph.update_all(msg_fn, _lstm_reducer))


# lstm_dgl()
# exit()


def run_lstm(module, count):
    cnt = 0
    cnt2 = 0
    num_call = 0
    overall_time = 0
    accumulate = 0
    for it, item in enumerate(count):
        if item > 0:
            accumulate += item
            if (it != 0 and it % 32
                    == 0) or (accumulate * it > 100000
                              and accumulate > 64) or (it == len(count) - 1):
                item = accumulate
                accumulate = 0
            else:
                continue
            t = torch.randn([item, it, in_feat], device=torch.device(0))
            # print(t.shape)
            other = (t.new_zeros([1, item,
                                  in_feat]), t.new_zeros([1, item, in_feat]))
            torch.cuda.synchronize()
            t0 = time.time()
            module(t, other)
            torch.cuda.synchronize()
            t1 = time.time()
            overall_time += t1 - t0
            print("time", t1 - t0, "overall_time", overall_time, "batch size",
                  item, "seqlen", it)
            cnt += item
            cnt2 += item * it
            num_call += 1
    print("cnt", cnt, "cnt2", cnt2)
    print("num_call", num_call)


# cxgc.prof("lstm", "arxiv", lambda: run_lstm(lstm_module, count))
# reorder the graph to group the same-degree nodes together
metric = torch.argsort(deg, descending=False)
ptr, idx = cxgc.reorder_by(ptr, idx, metric)
cxgc.prof("lstm neighbor op", "arxiv",
          lambda: cxgc.NeighborLstmOP(lstm_module, ptr, idx, x, count))
cxgc.prof(
    "lstm neighbor op", "arxiv",
    lambda: cxgc.NeighborLstmPadOP(lstm_module, ptr, idx, x, count, in_feat, 10000))
output1 = cxgc.NeighborLstmOP(lstm_module, ptr, idx, x, count)
output2 = cxgc.NeighborLstmPadOP(lstm_module, ptr, idx, x, count, in_feat, 10000)
cxgc.compare(output1, output2)
# exit()


def run_lstm_one_op():
    # input: [batch, seqlen, hidden]
    # assuming the seq length is 20
    input_data = torch.randn(
        [num_center, 20, in_feat],
        device=torch.device(0),
    )
    batch_size = input_data.shape[0]
    other = (input_data.new_zeros([1, batch_size, in_feat]),
             input_data.new_zeros([1, batch_size, in_feat]))
    lstm_module.reset_parameters()
    _, (rst, _) = lstm_module(input_data, other)

    cxgc.prof("lstm one NN op, seq==20", "",
              lambda: lstm_module(input_data, other))


'''
does not work, the torch official implementation does not support such variant length
'''


def run_padded_seq():
    arr = []
    cnt = 0
    for item in deg:
        arr.append(torch.randn([item, in_feat], device=torch.device(0)))
        cnt += 1
    padded_seq = torch.nn.utils.rnn.pad_sequence(arr)
    other = (input_data.new_zeros([1, cnt, in_feat]),
             input_data.new_zeros([1, cnt, in_feat]))
    cxgc.prof("lstm", "padded", lambda: lstm_module(padded_seq, other))


# run_padded_seq()

run_lstm(lstm_module, count)
