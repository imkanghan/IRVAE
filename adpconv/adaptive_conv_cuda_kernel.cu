#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define DIM0(TENSOR) ((TENSOR).x)
#define DIM1(TENSOR) ((TENSOR).y)
#define DIM2(TENSOR) ((TENSOR).z)
#define DIM3(TENSOR) ((TENSOR).w)

#define DIM3_INDEX(TENSOR, xx, yy, zz, ww) ((TENSOR)[((xx) * (TENSOR##_stride.x)) + ((yy) * (TENSOR##_stride.y)) + ((zz) * (TENSOR##_stride.z)) + ((ww) * (TENSOR##_stride.w))])

namespace{
template <typename scalar_t>
__global__ void adaptive_conv_cuda_forward_kernel(
    const size_t n, const size_t kernel_length,
    const scalar_t* input, const long4 input_size, const long4 input_stride,
    const scalar_t* kernel, const long4 kernel_size, const long4 kernel_stride,
    scalar_t* output, const long4 output_size, const long4 output_stride) {

    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= n) return;

    int batch   = ( index / DIM3(output_size) / DIM2(output_size) / DIM1(output_size)   ) % DIM0(output_size);
	int channel = ( index / DIM3(output_size) / DIM2(output_size)                       ) % DIM1(output_size);
	int y       = ( index / DIM3(output_size)                                           ) % DIM2(output_size);
	int x       = ( index                                                               ) % DIM3(output_size);

    scalar_t val = 0.0;

    for(int i = 0; i < kernel_length; i += 1){
        for(int j = 0; j < kernel_length; j += 1){
            val += DIM3_INDEX(input, batch, channel, y + i, x + j) * DIM3_INDEX(kernel, batch, i * kernel_length + j, y, x);
        }
    }

    output[index] = val;
}

template <typename scalar_t>
__global__ void adaptive_conv_kernel_cuda_backward_kernel(
    size_t n, size_t kernel_length, size_t num_channels,
    scalar_t* grad_output, const long4 grad_output_stride,
    scalar_t* input, const long4 input_stride,
    scalar_t* grad_kernel, const long4 grad_kernel_size, const long4 grad_kernel_stride) {

    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= n) return;

    int batch   = ( index / DIM3(grad_kernel_size) / DIM2(grad_kernel_size) / DIM1(grad_kernel_size)   ) % DIM0(grad_kernel_size);
	int channel = ( index / DIM3(grad_kernel_size) / DIM2(grad_kernel_size)                       ) % DIM1(grad_kernel_size);
	int y       = ( index / DIM3(grad_kernel_size)                                           ) % DIM2(grad_kernel_size);
	int x       = ( index                                                               ) % DIM3(grad_kernel_size);

    scalar_t val = 0.0;
    
    // three channel
    for(int c = 0; c < num_channels; c += 1){
        val += DIM3_INDEX(grad_output, batch, c, y, x) 
            * DIM3_INDEX(input, batch, c, y + channel / kernel_length, x  + channel % kernel_length);
    }

    grad_kernel[index] = val;

}

/*
template <typename scalar_t>
__global__ void adaptive_conv_input_backward(
    size_t n, size_t kernel_length,
    scalar_t* grad_output, const long4 grad_output_stride,
    scalar_t* kernel, const long4 kernel_stride,
    scalar_t* grad_input, const long4 grad_input_size, const long4 grad_input_stride) {

    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= n) return;

    int batch   = ( index / DIM3(grad_input_size) / DIM2(grad_input_size) / DIM1(grad_input_size)   ) % DIM0(grad_input_size);
	int channel = ( index / DIM3(grad_input_size) / DIM2(grad_input_size)                           ) % DIM1(grad_input_size);
	int y       = ( index / DIM3(grad_input_size)                                                   ) % DIM2(grad_input_size);
	int x       = ( index                                                                           ) % DIM3(grad_input_size);

    scalar_t val = 0.0;
    
    // three channel
    for(int c = 0; c < 3; c += 1){
        val += DIM3_INDEX(grad_output, batch, c, y + channel / kernel_length, x + channel % kernel_length) 
            * DIM3_INDEX(input, batch, c, y + channel / kernel_length, x  + channel % kernel_length);
    }

    grad_input[index] = val;

}

*/

}//namespace


void adaptive_conv_cuda_forward(
    at::Tensor input,
    at::Tensor kernel,
    at::Tensor output) {

    const auto n = output.numel();
    const auto kernel_length = size_t(sqrt(kernel.size(1)));

    const long4 input_size = make_long4(input.size(0), input.size(1), input.size(2), input.size(3));
    const long4 kernel_size = make_long4(kernel.size(0), kernel.size(1), kernel.size(2), kernel.size(3));
    const long4 output_size = make_long4(output.size(0), output.size(1), output.size(2), output.size(3));

    const long4 input_stride    = make_long4(input.stride(0), input.stride(1), input.stride(2), input.stride(3));
    const long4 kernel_stride   = make_long4(kernel.stride(0), kernel.stride(1), kernel.stride(2), kernel.stride(3));
    const long4 output_stride      = make_long4(output.stride(0), output.stride(1), output.stride(2), output.stride(3));

    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(output.type(), "add_forward_cuda", ([&] {
    adaptive_conv_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        n, kernel_length,
        input.data<scalar_t>(), input_size, input_stride,
        kernel.data<scalar_t>(), kernel_size, kernel_stride,
        output.data<scalar_t>(), output_size, output_stride
        );
    }));

}

void adaptive_conv_cuda_backward(
    at::Tensor grad_output,
    at::Tensor input,
    at::Tensor kernel,
    at::Tensor grad_input,
    at::Tensor grad_kernel) {

    const auto n = grad_kernel.numel();
    const auto kernel_length = size_t(sqrt(grad_kernel.size(1)));
    const auto num_channels = input.size(1);

    const long4 grad_kernel_size  = make_long4(grad_kernel.size(0), grad_kernel.size(1), grad_kernel.size(2), grad_kernel.size(3));

    const long4 grad_output_stride  = make_long4(grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3));
    const long4 input_stride        = make_long4(input.stride(0), input.stride(1), input.stride(2), input.stride(3));
    const long4 grad_kernel_stride  = make_long4(grad_kernel.stride(0), grad_kernel.stride(1), grad_kernel.stride(2), grad_kernel.stride(3));


    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(kernel.type(), "add_backward_cuda", ([&] {
    adaptive_conv_kernel_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        n, kernel_length, num_channels,
        grad_output.data<scalar_t>(), grad_output_stride,
        input.data<scalar_t>(), input_stride,
        grad_kernel.data<scalar_t>(), grad_kernel_size, grad_kernel_stride);
    }));

}
