from .codegen.triton_typed_matmul import typed_matmul
from .codegen.util import prof
import torch
import cxgnncomp_backend
import torch.nn.functional as F
import time


# OOM
def TypedLinearNaiveS2D(x, weights, types, src, dst, num_center, num_edge):
    assert (num_edge != 0)
    scattered_feature = torch.index_select(x, dim=0, index=src[:num_edge])
    scattered_weight = torch.index_select(weights,
                                          dim=0,
                                          index=types[:num_edge])
    transformed_feat = torch.mm(scattered_feature, scattered_weight)
    output = torch.zeros([num_center, transformed_feat.shape[1]],
                         device=x.device,
                         dtype=x.dtype)
    output.index_add_(0, dst[:num_edge], transformed_feat)
    return output


def SelectMMS2EOP(x, weights, idx, types):
    output = []
    for i in range(weights.shape[0]):
        output.append(torch.mm(x[idx[types == i]], weights[i]))
    return output


def TypedLinearS2DPushOP(x, weights, types, src, dst, num_type, num_center,
                         num_edge):
    torch.cuda.synchronize()
    t0 = time.time()
    max_src = torch.max(src) + 1
    src_type = types * max_src + src
    sorted, indices = torch.sort(src_type)
    torch.cuda.synchronize()
    t0 = time.time()
    src_type, reverse_indices = torch.unique(src_type, return_inverse=True)
    torch.cuda.synchronize()
    t1 = time.time()
    type_unique = torch.div(src_type, max_src, rounding_mode='floor')
    src_unique = src_type % max_src
    new_dst = dst[indices]
    count = torch.bincount(reverse_indices)
    torch.cuda.synchronize()
    t2 = time.time()
    num_item = src_unique.shape[0]
    output = typed_matmul(x,
                          weights,
                          type_unique,
                          num_item,
                          src_unique,
                          seq_output=True)
    torch.cuda.synchronize()
    t3 = time.time()
    output = torch.repeat_interleave(output, count, dim=0)
    torch.cuda.synchronize()
    t4 = time.time()
    output_dst = torch.zeros([num_center, weights.shape[-1]],
                             dtype=x.dtype,
                             device=x.device)
    output_dst.index_add_(0, new_dst, output)
    torch.cuda.synchronize()
    t5 = time.time()
    print(t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)
    return output_dst


class TypedLinearE2EOP(torch.autograd.Function):
    # x: [E, dim]

    @staticmethod
    def preprocess(weights, types, thres):
        num_rel = weights.shape[0]
        num_item = types.shape[0]
        count = torch.bincount(types)
        new_types = torch.empty([types.shape[0] + num_rel * (thres - 1)],
                                device=weights.device,
                                dtype=types.dtype)
        new_types[:types.shape[0]] = types
        cxgnncomp_backend.pad_rel(new_types, count, thres, num_rel,
                                  types.shape[0])
        sorted_types, indices = torch.sort(new_types)
        return [sorted_types, num_item, indices]

    @staticmethod
    def forward(ctx,
                x,
                weights,
                types,
                preprocessed=[],
                thres=256,
                count=None):
        if len(preprocessed) == 3:
            # return typed_matmul(x, weights, types, x.shape[0])
            sorted_types, num_item, indices = preprocessed
        else:
            sorted_types, num_item, indices = TypedLinearE2EOP.preprocess(
                weights, types, thres)
        # print(torch.max(indices), x.shape, sorted_types.shape, indices.shape)
        output = typed_matmul(x, weights, sorted_types, num_item, indices)
        ctx.save_for_backward(x, weights, types, indices, sorted_types)
        return output[:num_item]

    @staticmethod
    def backward(ctx, grad_out):
        # a naive implementation
        x, weights, types, indices, sorted_types = ctx.saved_tensors
        num_rel = weights.shape[0]
        num_item = types.shape[0]
        trans_weights = torch.transpose(weights, 1, 2).contiguous()
        grad_x = typed_matmul(grad_out, trans_weights, sorted_types, num_item,
                              indices)
        grad_weights = torch.empty_like(weights)
        for i in range(num_rel):
            # the indexing here is not efficient
            grad_weights[i] = torch.mm(x[types == i].t(), grad_out[types == i])
        return grad_x, grad_weights, None


class TypedLinearS2EOP(torch.autograd.Function):
    # x: [Src, dim]

    @staticmethod
    def preprocess(weights, types, src_idx, thres):
        num_rel = weights.shape[0]
        num_item = src_idx.shape[0]
        count = torch.bincount(types)
        new_types = torch.empty([types.shape[0] + num_rel * (thres - 1)],
                                device=weights.device,
                                dtype=types.dtype)
        new_types[:types.shape[0]] = types
        new_src_idx = torch.empty([src_idx.shape[0] + num_rel * (thres - 1)],
                                  device=weights.device,
                                  dtype=src_idx.dtype)
        new_src_idx[:src_idx.shape[0]] = src_idx
        cxgnncomp_backend.pad_rel_idx(new_types, new_src_idx, count, thres,
                                      num_rel, types.shape[0])
        sorted_types, indices = torch.sort(new_types)
        print("num valid item for S2E", num_item)
        return [sorted_types, num_item, new_src_idx, indices]

    @staticmethod
    def forward(ctx,
                x,
                weights,
                types,
                src_idx,
                preprocessed=[],
                thres=256,
                count=None):
        ctx.save_for_backward(x, weights, types, src_idx)
        if len(preprocessed) == 4:
            sorted_types, num_item, new_src_idx, indices = preprocessed
        else:
            sorted_types, num_item, new_src_idx, indices = TypedLinearS2EOP.preprocess(
                x, weights, types, src_idx, thres)
        output = typed_matmul(
            x,
            weights,
            sorted_types,
            num_item,  # FIXME: num_item may be incorrect
            new_src_idx[indices],
            indices)
        return output[:num_item]

    @staticmethod
    def backward(ctx, grad_out):
        pass


class TypedLinearS2DMMAggrOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weights, ptr, idx, rel, num_center):
        ctx.save_for_backward(x, weights, ptr, idx, rel)
        num_rel = weights.shape[0]
        output = torch.zeros([num_center, weights.shape[-1]], device=x.device)
        for i in range(num_rel):
            transformed_x = torch.mm(x, weights[i])
            cxgnncomp_backend.selective_aggr(transformed_x, ptr, idx,
                                             (rel == i), output, num_center)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        x, weights, ptr, idx, rel = ctx.saved_tensors
        num_rel = weights.shape[0]
        grad_x = torch.zeros_like(x)
        grad_weights = []
        num_center = grad_out.shape[0]
        num_node = x.shape[0]
        x_t = x.transpose(0, 1)
        for i in range(num_rel):
            grad_selective = torch.zeros([num_node, grad_out.shape[-1]],
                                         device=x.device)
            cxgnncomp_backend.selective_aggr_bwd(
                grad_out, ptr, idx, (rel == i), grad_selective,
                num_center)  # pass grad through selective_aggr
            grad_x += torch.mm(grad_selective, weights[i].transpose(0, 1))
            grad_weights.append(torch.mm(x_t, grad_selective))
        return grad_x, torch.stack(grad_weights), None, None, None, None


class TypedLinearS2DAggrMMOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weights, ptr, idx, rel, num_center):
        num_rel = weights.shape[0]
        output = torch.zeros([num_center, weights.shape[-1]], device=x.device)
        aggr_outputs = []
        for i in range(num_rel):
            aggr_output = torch.zeros([num_center, weights.shape[-2]],
                                      device=x.device)
            cxgnncomp_backend.selective_aggr(x, ptr, idx, (rel == i),
                                             aggr_output, num_center)
            output += torch.mm(aggr_output, weights[i])
            # output += torch.empty([num_center, weights.shape[-1]],
            #                       device=x.device)
            aggr_outputs.append(aggr_output)
        ctx.save_for_backward(x, weights, ptr, idx, rel,
                              torch.stack(aggr_outputs))
        return output

    @staticmethod
    def backward(ctx, grad_out):
        x, weights, ptr, idx, rel, aggr_outputs = ctx.saved_tensors
        num_rel = weights.shape[0]
        grad_x = torch.zeros_like(x)
        grad_weights = []
        num_center = grad_out.shape[0]
        for i in range(num_rel):
            grad_mm = torch.mm(grad_out, weights[i].transpose(0, 1))
            grad_weights.append(
                torch.mm(aggr_outputs[i].transpose(0, 1), grad_out))
            cxgnncomp_backend.selective_aggr_bwd(
                grad_mm, ptr, idx, (rel == i), grad_x,
                num_center)  # pass grad through selective_aggr
        return grad_x, torch.stack(grad_weights), None, None, None, None


# it can auto diff
def TypedLinearS2DSort(x, weights, ptr, idx, rel, num_center, count=None):
    torch.cuda.synchronize()
    t0 = time.time()
    dst = torch.repeat_interleave(torch.arange(num_center, device=x.device),
                                  ptr[1:num_center + 1] - ptr[:num_center])
    num_rel = weights.shape[0]
    output = torch.zeros([num_center, weights.shape[-1]], device=x.device)
    tgraph = 0
    tnn = 0
    tother = 0
    sorted_rel, indices = torch.sort(rel)
    torch.cuda.synchronize()
    t1 = time.time()
    tother = t1 - t0
    cnt = 0
    if count is None:
        count = torch.bincount(rel, minlength=num_rel).cpu()
    else:
        count = count.cpu()
    torch.cuda.synchronize()
    t0 = time.time()
    src = idx[indices]
    dst = dst[indices]
    torch.cuda.synchronize()
    t1 = time.time()
    tgraph += t1 - t0
    for i in range(num_rel):
        s = src[cnt:cnt + count[i]]
        d = dst[cnt:cnt + count[i]]
        cnt += count[i]
        torch.cuda.synchronize()
        t0 = time.time()
        feat = x[s]
        torch.cuda.synchronize()
        t1 = time.time()
        tgraph += t1 - t0
        transformed_feat = F.linear(feat, weights[i].T)
        torch.cuda.synchronize()
        t2 = time.time()
        output.index_add_(0, d, transformed_feat)
        torch.cuda.synchronize()
        t3 = time.time()
        tgraph += t3 - t2
        tnn += t2 - t1
    # print('graph time: {:3f}'.format(tgraph))
    # print('nn time: {:3f}'.format(tnn))
    # print('other time: {:3f}'.format(tother))
    # return output, tgraph, tnn, tother
    return output  # , tgraph, tnn, tother
