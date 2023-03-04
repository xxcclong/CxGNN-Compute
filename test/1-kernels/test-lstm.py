import torch
import cxgnncomp as cxgc
import time

in_feat = 256


def prepare_data():
    dset = "arxiv"
    num_head = 1
    x, ptr, idx, b = cxgc.prepare_data_full_graph(
        dset,
        feat_len=in_feat,
        num_head=num_head,
    )
    # x, ptr, idx, b = cxgc.prepare_data_sampled_graph(dset=dset,
    #                                                  feat_len=in_feat,
    #                                                  num_head=num_head,
    #                                                  num_seeds=1000)
    return x, ptr, idx, b, num_head


x, ptr, idx, batch, num_head = prepare_data()
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


def run_lstm_one_op():
    # input: [batch, seqlen, hidden]
    input_data = torch.randn(
        [num_center, 20, in_feat],
        device=torch.device(0),
    )
    batch_size = input_data.shape[0]
    other = (input_data.new_zeros([1, batch_size, in_feat]),
             input_data.new_zeros([1, batch_size, in_feat]))
    lstm_module.reset_parameters()
    _, (rst, _) = lstm_module(input_data, other)

    cxgc.prof("lstm", "", lambda: lstm_module(input_data, other))


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


cxgc.prof("lstm", "arxiv", lambda: run_lstm(lstm_module, count))
# run_padded_seq()

# run_lstm(lstm_module, count)