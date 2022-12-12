#include <torch/extension.h>
#include <vector>

void adaptive_conv_cuda_forward(
    at::Tensor input,
    at::Tensor kernel,
    at::Tensor output);

void adaptive_conv_cuda_backward(
    at::Tensor grad_output,
    at::Tensor input,
    at::Tensor kernel,
    at::Tensor grad_input,
    at::Tensor grad_kernel);


void adaptive_conv_forward(
    at::Tensor input,
    at::Tensor kernel,
    at::Tensor output) {

    adaptive_conv_cuda_forward(input, kernel, output);
}

void adaptive_conv_backward(
    at::Tensor grad_output,
    at::Tensor input,
    at::Tensor kernel,
    at::Tensor grad_input,
    at::Tensor grad_kernel) {
    
    adaptive_conv_cuda_backward(grad_output, input, kernel, grad_input, grad_kernel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &adaptive_conv_forward, "Adaptive convolution forward");
  m.def("backward", &adaptive_conv_backward, "Adaptive convolution backward");
}
