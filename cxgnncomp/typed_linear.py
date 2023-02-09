from .codegen.triton_typed_matmul import typed_matmul
import torch
import cxgnncomp_backend


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
    def forward(ctx, x, weights, types, src_idx, preprocessed=False, count=None):
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
        new_src_idx = torch.empty(
            [src_idx.shape[0] + num_rel * (thres - 1)],
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