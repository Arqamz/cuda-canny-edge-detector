#include "kernel.cuh"

// Basic CUDA kernel that prints "Hello World"
__global__ void helloWorldKernel() {
    printf("Hello World from thread [%d, %d]!\n", threadIdx.x, blockIdx.x);
}

// Host function to launch the kernel
void launchHelloWorldKernel() {
    // Launch the kernel with 1 block of 256 threads
    helloWorldKernel<<<1, 256>>>();
    
    // Synchronize to make sure the kernel completes
    cudaDeviceSynchronize();
    
    // Check for any errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}
