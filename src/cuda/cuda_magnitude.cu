#include "cuda_magnitude.cuh"

__global__ void magnitude_kernel(short int *delta_x, short int *delta_y, short int *magnitude, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols) {
        int pos = r * cols + c;
        int dx = (int)delta_x[pos];
        int dy = (int)delta_y[pos];
        int sum = dx * dx + dy * dy;
        magnitude[pos] = (short)(0.5f + sqrtf((float)sum));
    }
}

void cuda_magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols, short int **magnitude) {
    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    float gpu_time;

    // Allocate host memory for magnitude
    *magnitude = (short int *)malloc(rows * cols * sizeof(short int));
    if (*magnitude == NULL) {
        fprintf(stderr, "Error allocating the magnitude image on host.\n");
        return;
    }

    // Create CUDA events
    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate for start failed: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }
    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate for stop failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaEventDestroy(start);
        return;
    }

    size_t size = rows * cols * sizeof(short int);
    short int *d_delta_x = NULL;
    short int *d_delta_y = NULL;
    short int *d_magnitude = NULL;

    // Allocate device memory for delta_x
    cudaEventRecord(start);
    cudaStatus = cudaMalloc((void**)&d_delta_x, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc d_delta_x: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_delta_x failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    // Allocate device memory for delta_y
    cudaEventRecord(start);
    cudaStatus = cudaMalloc((void**)&d_delta_y, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc d_delta_y: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_delta_y failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_delta_x);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    // Allocate device memory for magnitude
    cudaEventRecord(start);
    cudaStatus = cudaMalloc((void**)&d_magnitude, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc d_magnitude: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_magnitude failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    // Copy delta_x and delta_y from host to device
    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(d_delta_x, delta_x, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy HtoD delta_x: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy delta_x HtoD failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaFree(d_magnitude);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(d_delta_y, delta_y, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy HtoD delta_y: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy delta_y HtoD failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaFree(d_magnitude);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    // Configure kernel launch parameters
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    // Launch kernel
    cudaEventRecord(start);
    magnitude_kernel<<<grid, block>>>(d_delta_x, d_delta_y, d_magnitude, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Kernel execution: %.2f ms\n", gpu_time);

    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaFree(d_magnitude);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    // Check for kernel execution errors
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaFree(d_magnitude);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    // Copy result from device to host
    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(*magnitude, d_magnitude, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy DtoH magnitude: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy magnitude DtoH failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaFree(d_magnitude);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    // Cleanup
    cudaFree(d_delta_x);
    cudaFree(d_delta_y);
    cudaFree(d_magnitude);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}