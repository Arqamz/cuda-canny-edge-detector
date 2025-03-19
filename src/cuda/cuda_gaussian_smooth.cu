#include "cuda_gaussian_smooth.cuh"

// Optimized kernel for horizontal Gaussian smoothing using shared memory
__global__ void gaussian_smooth_x_kernel(const unsigned char *d_image, float *d_temp, int rows, int cols, int kernel_radius) {
    
    extern __shared__ unsigned char s_data_x[];

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int s_r = threadIdx.y;
    int s_width = blockDim.x + 2 * kernel_radius;

    int global_c_start = blockIdx.x * blockDim.x - kernel_radius;
    int elements_per_thread = (s_width + blockDim.x - 1) / blockDim.x;

    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = threadIdx.x + i * blockDim.x;
        if (idx >= s_width) break;

        int load_c = global_c_start + idx;
        int s_index = s_r * s_width + idx;

        if (r < rows) {
            s_data_x[s_index] = (load_c >= 0 && load_c < cols) ? d_image[r * cols + load_c] : 0;
        } else {
            s_data_x[s_index] = 0;
        }
    }

    __syncthreads();

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) {
        float dot = 0.0f;
        #pragma unroll
        for (int i = -kernel_radius; i <= kernel_radius; i++) {
            int s_c = threadIdx.x + kernel_radius + i;
            dot += s_data_x[s_r * s_width + s_c] * d_gaussian_constants.kernel[kernel_radius + i];
        }
        d_temp[r * cols + c] = dot / d_gaussian_constants.sum;
    }
}

// Optimized kernel for vertical Gaussian smoothing using shared memory
__global__ void gaussian_smooth_y_kernel(const float *d_temp, short int *d_smoothed, int rows, int cols, int kernel_radius) {
    
    extern __shared__ float s_data_y[];

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int s_c = threadIdx.x;
    int s_height = blockDim.y + 2 * kernel_radius;

    int global_r_start = blockIdx.y * blockDim.y - kernel_radius;
    int elements_per_thread = (s_height + blockDim.y - 1) / blockDim.y;

    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = threadIdx.y + i * blockDim.y;
        if (idx >= s_height) break;

        int load_r = global_r_start + idx;
        int s_index = idx * blockDim.x + s_c;

        if (c < cols) {
            s_data_y[s_index] = (load_r >= 0 && load_r < rows) ? d_temp[load_r * cols + c] : 0.0f;
        } else {
            s_data_y[s_index] = 0.0f;
        }
    }

    __syncthreads();

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols) {
        float dot = 0.0f;
        #pragma unroll
        for (int i = -kernel_radius; i <= kernel_radius; i++) {
            int s_r = threadIdx.y + kernel_radius + i;
            dot += s_data_y[s_r * blockDim.x + s_c] * d_gaussian_constants.kernel[kernel_radius + i];
        }
        d_smoothed[r * cols + c] = (short int)(dot * 90.0f / d_gaussian_constants.sum + 0.5f);
    }
}

// Function to create Gaussian kernel and call GPU kernels
void cuda_gaussian_smooth(unsigned char *image, int rows, int cols, float sigma, short int **smoothedim)
{
    
    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    float gpu_time = 0.0f;

    // Create CUDA events
    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }
    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaEventDestroy(start);
        return;
    }

    int windowsize, center;
    float *kernel;

    struct {
        float kernel[MAX_KERNEL_SIZE];
        float sum;
    } h_gaussian_constants;

    // Create the Gaussian kernel
    make_gaussian_kernel(sigma, &kernel, &windowsize);
    if (windowsize > MAX_KERNEL_SIZE) {
        fprintf(stderr, "Error: Kernel size %d exceeds maximum allowed size %d\n", windowsize, MAX_KERNEL_SIZE);
        free(kernel);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        exit(1);
    }
    center = windowsize / 2;

    // Calculate kernel sum
    float kernel_sum = 0.0f;
    for (int i = 0; i < windowsize; i++) {
        kernel_sum += kernel[i];
    }

    memcpy(h_gaussian_constants.kernel, kernel, windowsize * sizeof(float));
    h_gaussian_constants.sum = kernel_sum;

    // Copy kernel struct to constant memory
    cudaEventRecord(start);
    cudaStatus = cudaMemcpyToSymbol(d_gaussian_constants, &h_gaussian_constants, sizeof(h_gaussian_constants));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpyToSymbol time: %.2f ms\n", gpu_time);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(cudaStatus));
        free(kernel);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        exit(1);
    }

    free(kernel);

    size_t imageSize = rows * cols * sizeof(unsigned char);
    size_t tempSize = rows * cols * sizeof(float);
    size_t smoothSize = rows * cols * sizeof(short int);

    unsigned char *d_image;
    float *d_temp;
    short int *d_smoothed;

    // Allocate device memory
    cudaEventRecord(start);
    cudaStatus = cudaMalloc((void **)&d_image, imageSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc d_image time: %.2f ms\n", gpu_time);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_image failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(1);
    }

    cudaEventRecord(start);
    cudaStatus = cudaMalloc((void **)&d_temp, tempSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc d_temp time: %.2f ms\n", gpu_time);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_temp failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_image);
        exit(1);
    }

    cudaEventRecord(start);
    cudaStatus = cudaMalloc((void **)&d_smoothed, smoothSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc d_smoothed time: %.2f ms\n", gpu_time);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_smoothed failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_image);
        cudaFree(d_temp);
        exit(1);
    }

    // Copy input image to device
    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy to device time: %.2f ms\n", gpu_time);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_image failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_image);
        cudaFree(d_temp);
        cudaFree(d_smoothed);
        exit(1);
    }

    // Set up grid and block dimensions
    dim3 blockDim_x(32, 8);
    dim3 gridDim_x((cols + blockDim_x.x - 1) / blockDim_x.x, (rows + blockDim_x.y - 1) / blockDim_x.y);

    dim3 blockDim_y(8, 32);
    dim3 gridDim_y((cols + blockDim_y.x - 1) / blockDim_y.x, (rows + blockDim_y.y - 1) / blockDim_y.y);

    size_t x_shared = (blockDim_x.x + 2 * center) * blockDim_x.y * sizeof(unsigned char);
    size_t y_shared = (blockDim_y.y + 2 * center) * blockDim_y.x * sizeof(float);

    // Launch horizontal Gaussian kernel
    cudaEventRecord(start);
    gaussian_smooth_x_kernel<<<gridDim_x, blockDim_x, x_shared>>>(d_image, d_temp, rows, cols, center);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("gaussian_smooth_x_kernel execution time: %.2f ms\n", gpu_time);
    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "gaussian_smooth_x_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_image);
        cudaFree(d_temp);
        cudaFree(d_smoothed);
        exit(1);
    }

    // Launch vertical Gaussian kernel
    cudaEventRecord(start);
    gaussian_smooth_y_kernel<<<gridDim_y, blockDim_y, y_shared>>>(d_temp, d_smoothed, rows, cols, center);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("gaussian_smooth_y_kernel execution time: %.2f ms\n", gpu_time);
    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "gaussian_smooth_y_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_image);
        cudaFree(d_temp);
        cudaFree(d_smoothed);
        exit(1);
    }

    // Allocate and copy result back to host
    *smoothedim = (short int *)malloc(smoothSize);
    if (*smoothedim == NULL) {
        fprintf(stderr, "Error allocating the smoothedim array.\n");
        cudaFree(d_image);
        cudaFree(d_temp);
        cudaFree(d_smoothed);
        exit(1);
    }

    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(*smoothedim, d_smoothed, smoothSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy from device time: %.2f ms\n", gpu_time);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy smoothedim failed: %s\n", cudaGetErrorString(cudaStatus));
        free(*smoothedim);
        cudaFree(d_image);
        cudaFree(d_temp);
        cudaFree(d_smoothed);
        exit(1);
    }

    // Cleanup
    cudaFree(d_image);
    cudaFree(d_temp);
    cudaFree(d_smoothed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(1);
    }

}