#ifndef CUDA_KERNEL_CUH
#define CUDA_KERNEL_CUH

#include <stdio.h>
#include <cuda_runtime.h>

// Basic CUDA kernel that prints "Hello World"
__global__ void helloWorldKernel();

// Host function to launch the kernel
void launchHelloWorldKernel();

#endif  // CUDA_KERNEL_CUH