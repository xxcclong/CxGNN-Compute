import torch
import cxgnncomp_backend


class SpMMValOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, ptr, idx, val, num_center):
        ctx.save_for_backward(x, ptr, idx, val)
        output = cxgnncomp_backend.spmm_multihead(
            ptr, idx, val, x, num_center,
            cxgnncomp_backend.SPMM_MULTIHEAD_SCHEDULE.Naive)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        x, ptr, idx, val = ctx.saved_tensors
        num_center = grad_out.shape[0]
        num_edge = val.shape[0]

        grad_x = torch.zeros_like(x)
        cxgnncomp_backend.spmm_multihead_bwd(
            ptr,
            idx,
            val,
            grad_out,
            grad_x,
            grad_out.shape[0],
        )

        grad_val = torch.zeros_like(val)
        dst = torch.repeat_interleave(
            torch.arange(num_center, device=ptr.device),
            ptr[1:num_center + 1] - ptr[:num_center])
        cxgnncomp_backend.run_sddmm(idx, dst, x, grad_out, grad_val, num_edge)

        return grad_x, None, None, grad_val, None