from .codegen.util import prof
import time
import torch


def NeighborLstmOneByOneOP(module, ptr, idx, feat):
    num_center_node = ptr.shape[0] - 1
    feat_len = feat.shape[-1]
    output = torch.empty([num_center_node, feat.shape[-1]], device=feat.device)
    for i in range(ptr.shape[0] - 1):
        deg = ptr[i + 1] - ptr[i]
        if deg > 0:
            comp_idx = idx[ptr[i]:ptr[i + 1]]
            tmp = module(
                torch.index_select(feat, 0, comp_idx).view(deg, -1), None)
            output[i] = tmp[0].view(-1, feat_len).sum(0)
    return output


def NeighborLstmOP(module, ptr, idx, feat, deg_count):
    # check
    # assert not torch.any(deg[1:] < deg[:-1])  # increasing

    num_center_node = ptr.shape[0] - 1
    src_cnt = 0
    dst_cnt = 0
    output = torch.empty([num_center_node, feat.shape[-1]], device=feat.device)
    for deg, num in enumerate(deg_count):
        if num > 0 and deg > 0:
            # begin lstm for "num" center nodes and "num * deg" neighbors
            comp_idx = idx[src_cnt:src_cnt + num * deg]
            tmp = module(
                torch.index_select(feat, 0, comp_idx).view(num, deg, -1), None)
            output[dst_cnt:dst_cnt + num] = tmp[0].view(num, deg, -1).sum(1)
            src_cnt += num * deg
            dst_cnt += num
    return output


def NeighborLstmPadOP(module, ptr, idx, feat, deg_count, num_center_in_batch,
                      num_neighbor_in_batch):
    # check
    # assert not torch.any(deg[1:] < deg[:-1])  # increasing

    num_center_node = ptr.shape[0] - 1
    src_cnt = 0
    dst_cnt = 0
    accumulated_dst = 0
    accumulated_src = 0
    output = torch.empty([num_center_node, feat.shape[-1]], device=feat.device)
    for deg, num in enumerate(deg_count):
        if num > 0 and deg > 0:
            if accumulated_dst == 0:
                min_deg = deg
            accumulated_dst += num
            accumulated_src += num * deg
        if accumulated_dst >= num_center_in_batch or accumulated_src >= num_neighbor_in_batch or (
                deg == len(deg_count) - 1 and accumulated_dst > 0):
            max_deg = deg
            comp_in = torch.zeros([accumulated_dst, max_deg, feat.shape[-1]],
                                  device=feat.device)
            tmp_dst_cnt = 0
            for deg2 in range(min_deg, max_deg + 1):
                num2 = deg_count[deg2]
                if num2 > 0:
                    comp_idx = idx[src_cnt:src_cnt + num2 * deg2]
                    comp_in[tmp_dst_cnt:tmp_dst_cnt +
                            num2, :deg2, :] = torch.index_select(
                                feat, 0, comp_idx).view(num2, deg2, -1)
                    src_cnt += num2 * deg2
                    tmp_dst_cnt += num2
            # torch.cuda.synchronize()
            # t0 = time.time()
            tmp = module(comp_in, None)
            # torch.cuda.synchronize()
            # t1 = time.time()
            # print(
            #     f"time {t1 - t0} for {accumulated_dst} nodes, deg {max_deg}, edges {accumulated_src}, {(t1 - t0) / accumulated_src}, comp_in {comp_in.shape}"
            # )
            # print(tmp[0].shape, dst_cnt, output.shape)
            output[dst_cnt:dst_cnt + accumulated_dst] = tmp[0].view(
                accumulated_dst, max_deg, -1).sum(1)
            dst_cnt += accumulated_dst
            # clear accumulated
            accumulated_dst = 0
            accumulated_src = 0
    return output