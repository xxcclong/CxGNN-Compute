from .codegen.triton_typed_matmul import typed_matmul
import torch
import cxgnncomp_backend
import torch.nn.functional as F


class TypedLinearE2EOP(torch.autograd.Function):
    # x: [E, dim]

    @staticmethod
    def forward(ctx, x, weights, types, preprocessed=False, count=None):
        ctx.save_for_backward(x, weights, types)
        if preprocessed:
            return typed_matmul(x, weights, types, x.shape[0])
        thres = 256
        num_rel = weights.shape[0]
        num_item = types.shape[0]
        if count is None:
            count = torch.bincount(types)
        else:
            count = count.to(x.device)
        new_types = torch.empty([types.shape[0] + num_rel * (thres - 1)],
                                device=x.device,
                                dtype=types.dtype)
        new_types[:types.shape[0]] = types
        cxgnncomp_backend.pad_rel(new_types, count, thres, num_rel,
                                  types.shape[0])
        sorted_types, indices = torch.sort(new_types)
        output = typed_matmul(x, weights, sorted_types, num_item, indices)
        return output[:num_item]

    @staticmethod
    def backward(ctx, grad_out):
        pass


class TypedLinearS2EOP(torch.autograd.Function):
    # x: [Src, dim]

    @staticmethod
    def forward(ctx,
                x,
                weights,
                types,
                src_idx,
                preprocessed=False,
                count=None):
        ctx.save_for_backward(x, weights, types, src_idx)
        if preprocessed:
            return typed_matmul(x, weights, types, src_idx, types.shape[0])
        thres = 256
        num_rel = weights.shape[0]
        num_item = types.shape[0]
        if count is None:
            count = torch.bincount(types)
        else:
            count = count.to(x.device)
        new_types = torch.empty([types.shape[0] + num_rel * (thres - 1)],
                                device=x.device,
                                dtype=types.dtype)
        new_types[:types.shape[0]] = types
        new_src_idx = torch.empty([src_idx.shape[0] + num_rel * (thres - 1)],
                                  device=x.device,
                                  dtype=src_idx.dtype)
        new_src_idx[:src_idx.shape[0]] = src_idx
        cxgnncomp_backend.pad_rel_idx(new_types, new_src_idx, count, thres,
                                      num_rel, types.shape[0])
        sorted_types, indices = torch.sort(new_types)
        output = typed_matmul(
            x,
            weights,
            sorted_types,
            num_item,
            new_src_idx[indices],
            indices,
        )
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
    dst = torch.repeat_interleave(torch.arange(num_center, device=x.device),
                                  ptr[1:num_center + 1] - ptr[:num_center])
    num_rel = weights.shape[0]
    output = torch.zeros([num_center, weights.shape[-1]], device=x.device)
    sorted_rel, indices = torch.sort(rel)
    cnt = 0
    if count is None:
        count = torch.bincount(rel, minlength=num_rel).cpu()
    else:
        count = count.cpu()
    src = idx[indices]
    dst = dst[indices]
    for i in range(num_rel):
        s = src[cnt:cnt + count[i]]
        d = dst[cnt:cnt + count[i]]
        cnt += count[i]
        feat = x[s]
        transformed_feat = F.linear(feat, weights[i].T)
        output.index_add_(0, d, transformed_feat)
    return output
