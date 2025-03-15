#include <stdio.h>

#include "kernel.cuh"

int main()
{
    printf("Hello World from CPU!\n");

    launchHelloWorldKernel();
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    
    printf("GPU computation completed successfully!\n");
    
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    return 0;
}
