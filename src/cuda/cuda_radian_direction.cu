#include "cuda_radian_direction.cuh"

__global__ void compute_radian_direction_kernel(const short int* delta_x, const short int* delta_y, float* dir_radians, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;

    int pos = r * cols + c;

    float dx = static_cast<float>(delta_x[pos]);
    float dy = static_cast<float>(delta_y[pos]);

    // Apply sign based on constant tags
    if (c_xdirtag == 1) dx = -dx;
    if (c_ydirtag == -1) dy = -dy;

    if (dx == 0.0f && dy == 0.0f) {
        dir_radians[pos] = 0.0f;
        return;
    }

    float ang = atan2f(dy, dx);
    if (ang < 0.0f) {
        ang += 2.0f * CUDART_PI_F;
    }

    dir_radians[pos] = ang;
}

void cuda_radian_direction(short int *delta_x, short int *delta_y, int rows, int cols, float **dir_radians, int xdirtag, int ydirtag) {
    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    float gpu_time = 0.0f;

    // Create CUDA events for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate host memory for the direction image
    int num_pixels = rows * cols;
    float *dirim = (float *)malloc(num_pixels * sizeof(float));
    if (dirim == NULL) {
        fprintf(stderr, "Error allocating the gradient direction image.\n");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }
    *dir_radians = dirim;

    // Allocate device memory for delta_x, delta_y, and dir_radians
    short int *d_delta_x = NULL;
    short int *d_delta_y = NULL;
    float *d_dir_radians = NULL;
    size_t delta_size = num_pixels * sizeof(short int);
    size_t dir_size = num_pixels * sizeof(float);

    // Allocate d_delta_x
    cudaEventRecord(start);
    cudaStatus = cudaMalloc((void**)&d_delta_x, delta_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc d_delta_x: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_delta_x failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(dirim);
        return;
    }

    // Allocate d_delta_y
    cudaEventRecord(start);
    cudaStatus = cudaMalloc((void**)&d_delta_y, delta_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc d_delta_y: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_delta_y failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_delta_x);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(dirim);
        return;
    }

    // Allocate d_dir_radians
    cudaEventRecord(start);
    cudaStatus = cudaMalloc((void**)&d_dir_radians, dir_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc d_dir_radians: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_dir_radians failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(dirim);
        return;
    }

    // Copy xdirtag and ydirtag to constant memory
    cudaEventRecord(start);
    cudaStatus = cudaMemcpyToSymbol(c_xdirtag, &xdirtag, sizeof(int));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpyToSymbol c_xdirtag: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol c_xdirtag failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaFree(d_dir_radians);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(dirim);
        return;
    }

    cudaEventRecord(start);
    cudaStatus = cudaMemcpyToSymbol(c_ydirtag, &ydirtag, sizeof(int));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpyToSymbol c_ydirtag: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol c_ydirtag failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaFree(d_dir_radians);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(dirim);
        return;
    }

    // Copy input data from host to device
    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(d_delta_x, delta_x, delta_size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy delta_x to device: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy delta_x failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaFree(d_dir_radians);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(dirim);
        return;
    }

    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(d_delta_y, delta_y, delta_size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy delta_y to device: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy delta_y failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaFree(d_dir_radians);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(dirim);
        return;
    }

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    cudaEventRecord(start);
    compute_radian_direction_kernel<<<grid, block>>>(d_delta_x, d_delta_y, d_dir_radians, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Kernel execution time: %.2f ms\n", gpu_time);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaFree(d_dir_radians);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(dirim);
        return;
    }

    // Copy output from device to host
    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(dirim, d_dir_radians, dir_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy dir_radians to host: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy dir_radians failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaFree(d_dir_radians);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(dirim);
        return;
    }

    // Cleanup
    cudaFree(d_delta_x);
    cudaFree(d_delta_y);
    cudaFree(d_dir_radians);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}