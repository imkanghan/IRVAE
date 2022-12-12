import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load
import pdb

adaptive_conv_cuda = load('adaptive_conv_cuda',
        ['./adpconv/adaptive_conv_cuda.cpp', './adpconv/adaptive_conv_cuda_kernel.cu'],
        verbose=True)

class AdaptiveConvFunction(Function):
    def __init__(self):
        super(AdaptiveConvFunction, self).__init__()

    @staticmethod
    def forward(ctx, x, kernel):
        ctx.save_for_backward(x, kernel)

        output = kernel.new_zeros((x.size(0), x.size(1), kernel.size(2), kernel.size(3)))
        adaptive_conv_cuda.forward(x, kernel, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, kernel = ctx.saved_tensors
        grad_x = torch.zeros_like(x)
        grad_kernel = torch.zeros_like(kernel)

        adaptive_conv_cuda.backward(grad_output, x, kernel, grad_x, grad_kernel)

        return grad_x, grad_kernel


class AdaptiveConv(torch.nn.Module):
    def __init__(self):
        super(AdaptiveConv, self).__init__()

    def forward(self, x, kernel):
        return AdaptiveConvFunction.apply(x, kernel)


