#include "cuda_hysteresis.cuh"

// Constants for directional offsets
__constant__ int c_dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
__constant__ int c_dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

// Kernel for first step: marking initial edges above high threshold
__global__ void mark_initial_edges_kernel(short* mag, unsigned char* edge, int rows, int cols, int highthreshold) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (r < rows && c < cols) {
        int pos = r * cols + c;
        if (edge[pos] == POSSIBLE_EDGE && mag[pos] >= highthreshold) {
            edge[pos] = EDGE;
        }
    }
}

// Kernel for edge propagation (iterative)
__global__ void propagate_edges_kernel(unsigned char* edge, short* mag, int rows, int cols, int lowthreshold, int* changed) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    __shared__ int block_changed;
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        block_changed = 0;
    }
    __syncthreads();
    
    if (r < rows && c < cols) {
        int pos = r * cols + c;
        
        // Only POSSIBLE_EDGE pixels need to be checked
        if (edge[pos] == POSSIBLE_EDGE) {
            // Check all 8 neighbors
            for (int i = 0; i < 8; i++) {
                int nr = r + c_dy[i];
                int nc = c + c_dx[i];
                
                // Boundary check
                if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) {
                    continue;
                }
                
                int npos = nr * cols + nc;
                
                // If neighbor is an EDGE and this pixel is above low threshold
                if (edge[npos] == EDGE && mag[pos] >= lowthreshold) {
                    edge[pos] = EDGE;
                    atomicExch(&block_changed, 1);
                    break;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Only one thread per block updates the global flag
    if (threadIdx.x == 0 && threadIdx.y == 0 && block_changed) {
        atomicExch(changed, 1);
    }
}

// Kernel for final cleanup (set all non-EDGE to NOEDGE)
__global__ void cleanup_edges_kernel(unsigned char* edge, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (r < rows && c < cols) {
        int pos = r * cols + c;
        if (edge[pos] != EDGE) {
            edge[pos] = NOEDGE;
        }
    }
}

// Initialize edge map (border cleaning and marking possible edges)
__global__ void initialize_edge_map_kernel(unsigned char* nms, unsigned char* edge, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (r < rows && c < cols) {
        int pos = r * cols + c;
        
        // Mark as possible edge or not
        if (nms[pos] == POSSIBLE_EDGE) {
            edge[pos] = POSSIBLE_EDGE;
        } else {
            edge[pos] = NOEDGE;
        }
        
        // Clear borders
        if (r == 0 || r == rows-1 || c == 0 || c == cols-1) {
            edge[pos] = NOEDGE;
        }
    }
}

// Kernel to compute histogram
__global__ void compute_histogram_kernel(short* mag, unsigned char* edge, int rows, int cols, int* hist) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (r < rows && c < cols) {
        int pos = r * cols + c;
        if (edge[pos] == POSSIBLE_EDGE) {
            atomicAdd(&hist[mag[pos]], 1);
        }
    }
}

// Main CUDA implementation of hysteresis
cudaError_t cuda_apply_hysteresis(short* mag, unsigned char* nms, int rows, int cols, 
                                 float tlow, float thigh, unsigned char* edge) {
    cudaError_t cudaStatus;
    float gpu_time = 0.0f;
    cudaEvent_t start, stop;
    
    // Initialize CUDA events for timing
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Allocate device memory
    short* d_mag = nullptr;
    unsigned char* d_nms = nullptr;
    unsigned char* d_edge = nullptr;
    int* d_hist = nullptr;
    int* d_changed = nullptr;
    
    size_t image_size = rows * cols * sizeof(unsigned char);
    size_t mag_size = rows * cols * sizeof(short);
    size_t hist_size = 32768 * sizeof(int);
    
    printf("Allocating device memory...\n");
    
    // Allocate and copy magnitude data
    cudaEventRecord(start);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_mag, mag_size));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc magnitude: %.2f ms\n", gpu_time);
    
    cudaEventRecord(start);
    CHECK_CUDA_ERROR(cudaMemcpy(d_mag, mag, mag_size, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy HtoD magnitude: %.2f ms\n", gpu_time);
    
    // Allocate and copy non-maximal suppression data
    cudaEventRecord(start);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_nms, image_size));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc nms: %.2f ms\n", gpu_time);
    
    cudaEventRecord(start);
    CHECK_CUDA_ERROR(cudaMemcpy(d_nms, nms, image_size, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy HtoD nms: %.2f ms\n", gpu_time);
    
    // Allocate edge output buffer
    cudaEventRecord(start);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_edge, image_size));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc edge: %.2f ms\n", gpu_time);
    
    // Allocate histogram array
    cudaEventRecord(start);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_hist, hist_size));
    CHECK_CUDA_ERROR(cudaMemset(d_hist, 0, hist_size));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc and clear histogram: %.2f ms\n", gpu_time);
    
    // Allocate changed flag
    cudaEventRecord(start);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_changed, sizeof(int)));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc changed flag: %.2f ms\n", gpu_time);
    
    // Configure kernel dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                  (rows + blockSize.y - 1) / blockSize.y);
    
    // Step 1: Initialize edge map
    cudaEventRecord(start);
    initialize_edge_map_kernel<<<gridSize, blockSize>>>(d_nms, d_edge, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Initialize edge map kernel: %.2f ms\n", gpu_time);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Step 2: Compute histogram
    cudaEventRecord(start);
    compute_histogram_kernel<<<gridSize, blockSize>>>(d_mag, d_edge, rows, cols, d_hist);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Compute histogram kernel: %.2f ms\n", gpu_time);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy histogram back to host for threshold calculation
    int* h_hist = new int[32768];
    cudaEventRecord(start);
    CHECK_CUDA_ERROR(cudaMemcpy(h_hist, d_hist, hist_size, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy DtoH histogram: %.2f ms\n", gpu_time);
    
    // Calculate thresholds on CPU
    int numedges = 0;
    int maximum_mag = 0;
    
    for (int r = 1; r < 32768; r++) {
        if (h_hist[r] != 0) {
            maximum_mag = r;
        }
        numedges += h_hist[r];
    }
    
    int highcount = (int)(numedges * thigh + 0.5);
    
    int r = 1;
    numedges = h_hist[1];
    while ((r < (maximum_mag - 1)) && (numedges < highcount)) {
        r++;
        numedges += h_hist[r];
    }
    int highthreshold = r;
    int lowthreshold = (int)(highthreshold * tlow + 0.5);
    
    printf("Thresholds computed: low = %d, high = %d\n", lowthreshold, highthreshold);
    delete[] h_hist;
    
    // Step 3: Mark initial edges (pixels above high threshold)
    cudaEventRecord(start);
    mark_initial_edges_kernel<<<gridSize, blockSize>>>(d_mag, d_edge, rows, cols, highthreshold);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Mark initial edges kernel: %.2f ms\n", gpu_time);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Step 4: Iterative edge propagation (replaces recursive follow_edges)
    int max_iterations = 100;  // prevent infinite loops
    int iterations = 0;
    
    printf("Starting edge propagation...\n");
    
    do {
        // Reset change flag
        CHECK_CUDA_ERROR(cudaMemset(d_changed, 0, sizeof(int)));
        
        // Propagate edges
        cudaEventRecord(start);
        propagate_edges_kernel<<<gridSize, blockSize>>>(d_edge, d_mag, rows, cols, lowthreshold, d_changed);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time, start, stop);
        if (iterations == 0 || iterations == max_iterations-1) {
            printf("Edge propagation iteration %d: %.2f ms\n", iterations, gpu_time);
        }
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        int changed = 0;
        CHECK_CUDA_ERROR(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        
        if (!changed) {
            break;
        }
        
        iterations++;
    } while (iterations < max_iterations);
    
    printf("Edge propagation complete after %d iterations\n", iterations);
    
    // Step 5: Final cleanup (set all non-EDGE pixels to NOEDGE)
    cudaEventRecord(start);
    cleanup_edges_kernel<<<gridSize, blockSize>>>(d_edge, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Cleanup edges kernel: %.2f ms\n", gpu_time);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy result back to host
    cudaEventRecord(start);
    CHECK_CUDA_ERROR(cudaMemcpy(edge, d_edge, image_size, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy DtoH edge result: %.2f ms\n", gpu_time);
    
    // Free device memory
    cudaEventRecord(start);
    cudaFree(d_mag);
    cudaFree(d_nms);
    cudaFree(d_edge);
    cudaFree(d_hist);
    cudaFree(d_changed);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Free device memory: %.2f ms\n", gpu_time);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return cudaSuccess;
}