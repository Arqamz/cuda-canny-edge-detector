#include "cuda_derivative.cuh"

// X derivative kernel using shared memory for row data
__global__ void compute_delta_x_kernel(const short int *smoothedim, short int *delta_x, int rows, int cols) {
    extern __shared__ short int row_data[];
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        // Load entire row into shared memory
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            row_data[c] = smoothedim[row * cols + c];
        }
        __syncthreads();

        if (col < cols) {
            if (col == 0) {
                delta_x[row * cols + col] = row_data[col + 1] - row_data[col];
            } else if (col == cols - 1) {
                delta_x[row * cols + col] = row_data[col] - row_data[col - 1];
            } else {
                delta_x[row * cols + col] = row_data[col + 1] - row_data[col - 1];
            }
        }
    }
}

// Y derivative kernel using column-wise shared memory tiles with halo
__global__ void compute_delta_y_kernel(const short int *smoothedim, short int *delta_y, int rows, int cols) {
    extern __shared__ short int shared_col[];
    int col = blockIdx.x;
    int row_in_block = threadIdx.y;
    int tile_start_row = blockIdx.y * blockDim.y;

    // Load main tile elements
    int load_row = tile_start_row + row_in_block;
    if (load_row < rows && col < cols) {
        shared_col[row_in_block + 1] = smoothedim[load_row * cols + col];
    }

    // Load halo above
    if (row_in_block == 0) {
        int halo_row = tile_start_row - 1;
        if (halo_row < 0) halo_row = 0;
        shared_col[0] = smoothedim[halo_row * cols + col];
    }

    // Load halo below
    if (row_in_block == blockDim.y - 1) {
        int halo_row = tile_start_row + blockDim.y;
        if (halo_row >= rows) halo_row = rows - 1;
        shared_col[blockDim.y + 1] = smoothedim[halo_row * cols + col];
    }

    __syncthreads();

    int row = tile_start_row + row_in_block;
    if (row >= rows || col >= cols) return;

    if (row == 0) {
        delta_y[row * cols + col] = shared_col[1 + 1] - shared_col[1];
    } else if (row == rows - 1) {
        delta_y[row * cols + col] = shared_col[blockDim.y + 1] - shared_col[blockDim.y];
    } else {
        delta_y[row * cols + col] = shared_col[row_in_block + 2] - shared_col[row_in_block];
    }
}

void cuda_derivative_x_y(short int *smoothedim, int rows, int cols, short int **delta_x, short int **delta_y) {
    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    float gpu_time = 0.0f;

    short int *d_smoothed = NULL, *d_delta_x = NULL, *d_delta_y = NULL;
    size_t size = rows * cols * sizeof(short int);

    // Initialize CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate device memory
    cudaEventRecord(start);
    cudaStatus = cudaMalloc((void**)&d_smoothed, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc d_smoothed: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_smoothed failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    cudaEventRecord(start);
    cudaStatus = cudaMalloc((void**)&d_delta_x, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc d_delta_x: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_delta_x failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_smoothed);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    cudaEventRecord(start);
    cudaStatus = cudaMalloc((void**)&d_delta_y, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc d_delta_y: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_delta_y failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_smoothed);
        cudaFree(d_delta_x);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    // Copy input to device
    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(d_smoothed, smoothedim, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy to device: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_smoothed failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_smoothed);
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    // Launch delta_x kernel
    dim3 block_x(TILE_WIDTH, 1);
    dim3 grid_x((cols + TILE_WIDTH - 1) / TILE_WIDTH, rows);
    size_t shared_x = cols * sizeof(short);

    cudaEventRecord(start);
    compute_delta_x_kernel<<<grid_x, block_x, shared_x>>>(d_smoothed, d_delta_x, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("compute_delta_x_kernel: %.2f ms\n", gpu_time);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "delta_x kernel failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_smoothed);
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    // Launch delta_y kernel
    dim3 block_y(1, TILE_HEIGHT);
    dim3 grid_y(cols, (rows + TILE_HEIGHT - 1) / TILE_HEIGHT);
    size_t shared_y = (TILE_HEIGHT + 2) * sizeof(short);

    cudaEventRecord(start);
    compute_delta_y_kernel<<<grid_y, block_y, shared_y>>>(d_smoothed, d_delta_y, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("compute_delta_y_kernel: %.2f ms\n", gpu_time);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "delta_y kernel failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_smoothed);
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    // Allocate host memory
    *delta_x = (short int*)malloc(size);
    *delta_y = (short int*)malloc(size);
    if (!*delta_x || !*delta_y) {
        fprintf(stderr, "Host memory allocation failed\n");
        cudaFree(d_smoothed);
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    // Copy results back
    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(*delta_x, d_delta_x, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy delta_x: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy delta_x failed: %s\n", cudaGetErrorString(cudaStatus));
        free(*delta_x);
        free(*delta_y);
        cudaFree(d_smoothed);
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(*delta_y, d_delta_y, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy delta_y: %.2f ms\n", gpu_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy delta_y failed: %s\n", cudaGetErrorString(cudaStatus));
        free(*delta_x);
        free(*delta_y);
        cudaFree(d_smoothed);
        cudaFree(d_delta_x);
        cudaFree(d_delta_y);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    // Cleanup
    cudaFree(d_smoothed);
    cudaFree(d_delta_x);
    cudaFree(d_delta_y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}