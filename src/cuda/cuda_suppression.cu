#include "cuda_suppression.cuh"

#define NOEDGE 255
#define POSSIBLE_EDGE 128
#define EDGE 0

// Dimensions in constant memory
__constant__ int d_nrows;
__constant__ int d_ncols;

__global__ void non_max_supp_kernel(const short *mag, const short *gradx, const short *grady, unsigned char *result)
{
    extern __shared__ short shared_data[];
    
    short *s_mag = shared_data;
    short *s_gradx = &s_mag[(blockDim.y + 2) * (blockDim.x + 2)];
    short *s_grady = &s_gradx[(blockDim.y + 2) * (blockDim.x + 2)];
    
    // Global
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1; // +1 for boundary
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1; // +1 for boundary
    
    // Local with halo
    int s_row = threadIdx.y + 1;
    int s_col = threadIdx.x + 1;
    
    // Load into shared memory
    if (row < d_nrows - 1 && col < d_ncols - 1) {

        // Center pixels
        s_mag[s_row * (blockDim.x + 2) + s_col] = mag[row * d_ncols + col];
        s_gradx[s_row * (blockDim.x + 2) + s_col] = gradx[row * d_ncols + col];
        s_grady[s_row * (blockDim.x + 2) + s_col] = grady[row * d_ncols + col];
        
        // Load halo
        if (threadIdx.y == 0 && row > 1) {
            // Top row
            s_mag[0 * (blockDim.x + 2) + s_col] = mag[(row-1) * d_ncols + col];
            s_gradx[0 * (blockDim.x + 2) + s_col] = gradx[(row-1) * d_ncols + col];
            s_grady[0 * (blockDim.x + 2) + s_col] = grady[(row-1) * d_ncols + col];
        }
        
        if (threadIdx.y == blockDim.y - 1 && row < d_nrows - 2) {
            // Bottom row
            s_mag[(s_row+1) * (blockDim.x + 2) + s_col] = mag[(row+1) * d_ncols + col];
            s_gradx[(s_row+1) * (blockDim.x + 2) + s_col] = gradx[(row+1) * d_ncols + col];
            s_grady[(s_row+1) * (blockDim.x + 2) + s_col] = grady[(row+1) * d_ncols + col];
        }
        
        if (threadIdx.x == 0 && col > 1) {
            // Left column
            s_mag[s_row * (blockDim.x + 2) + 0] = mag[row * d_ncols + (col-1)];
            s_gradx[s_row * (blockDim.x + 2) + 0] = gradx[row * d_ncols + (col-1)];
            s_grady[s_row * (blockDim.x + 2) + 0] = grady[row * d_ncols + (col-1)];
        }
        
        if (threadIdx.x == blockDim.x - 1 && col < d_ncols - 2) {
            // Right column
            s_mag[s_row * (blockDim.x + 2) + (s_col+1)] = mag[row * d_ncols + (col+1)];
            s_gradx[s_row * (blockDim.x + 2) + (s_col+1)] = gradx[row * d_ncols + (col+1)];
            s_grady[s_row * (blockDim.x + 2) + (s_col+1)] = grady[row * d_ncols + (col+1)];
        }
        
        // Corners
        if (threadIdx.x == 0 && threadIdx.y == 0 && row > 1 && col > 1) {
            // Top-left
            s_mag[0 * (blockDim.x + 2) + 0] = mag[(row-1) * d_ncols + (col-1)];
            s_gradx[0 * (blockDim.x + 2) + 0] = gradx[(row-1) * d_ncols + (col-1)];
            s_grady[0 * (blockDim.x + 2) + 0] = grady[(row-1) * d_ncols + (col-1)];
        }
        
        if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0 && row > 1 && col < d_ncols - 2) {
            // Top-right
            s_mag[0 * (blockDim.x + 2) + (s_col+1)] = mag[(row-1) * d_ncols + (col+1)];
            s_gradx[0 * (blockDim.x + 2) + (s_col+1)] = gradx[(row-1) * d_ncols + (col+1)];
            s_grady[0 * (blockDim.x + 2) + (s_col+1)] = grady[(row-1) * d_ncols + (col+1)];
        }
        
        if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 && row < d_nrows - 2 && col > 1) {
            // Bottom-left
            s_mag[(s_row+1) * (blockDim.x + 2) + 0] = mag[(row+1) * d_ncols + (col-1)];
            s_gradx[(s_row+1) * (blockDim.x + 2) + 0] = gradx[(row+1) * d_ncols + (col-1)];
            s_grady[(s_row+1) * (blockDim.x + 2) + 0] = grady[(row+1) * d_ncols + (col-1)];
        }
        
        if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 && 
            row < d_nrows - 2 && col < d_ncols - 2) {
            // Bottom-right
            s_mag[(s_row+1) * (blockDim.x + 2) + (s_col+1)] = mag[(row+1) * d_ncols + (col+1)];
            s_gradx[(s_row+1) * (blockDim.x + 2) + (s_col+1)] = gradx[(row+1) * d_ncols + (col+1)];
            s_grady[(s_row+1) * (blockDim.x + 2) + (s_col+1)] = grady[(row+1) * d_ncols + (col+1)];
        }
    }
    
    __syncthreads();
    
    // Handle out-of-bounds threads
    if (row == 0 || col == 0 || row >= d_nrows - 1 || col >= d_ncols - 1) {
        if (row < d_nrows && col < d_ncols) {
            result[row * d_ncols + col] = (unsigned char)NOEDGE;
        }
        return;
    }
    
    if (row < d_nrows - 1 && col < d_ncols - 1 && row > 0 && col > 0) {
        short m00 = s_mag[s_row * (blockDim.x + 2) + s_col];
        
        if (m00 == 0) {
            result[row * d_ncols + col] = (unsigned char)NOEDGE;
            return;
        }
        
        short gx = s_gradx[s_row * (blockDim.x + 2) + s_col];
        short gy = s_grady[s_row * (blockDim.x + 2) + s_col];
        
        // Precalculate these values once
        float xperp = -gx / ((float)m00);
        float yperp = gy / ((float)m00);
        
        float mag1, mag2;
        short z1, z2;
        
        if (gx >= 0) {
            if (gy >= 0) {
            if (gx >= gy) {
                /* 111 */
                /* Left point */
                z1 = s_mag[s_row * (blockDim.x + 2) + (s_col - 1)];
                z2 = s_mag[(s_row - 1) * (blockDim.x + 2) + (s_col - 1)];
                
                mag1 = (m00 - z1) * xperp + (z2 - z1) * yperp;
                
                /* Right point */
                z1 = s_mag[s_row * (blockDim.x + 2) + (s_col + 1)];
                z2 = s_mag[(s_row + 1) * (blockDim.x + 2) + (s_col + 1)];
                
                mag2 = (m00 - z1) * xperp + (z2 - z1) * yperp;
            } else {
                /* 110 */
                /* Left point */
                z1 = s_mag[(s_row - 1) * (blockDim.x + 2) + s_col];
                z2 = s_mag[(s_row - 1) * (blockDim.x + 2) + (s_col - 1)];
                
                mag1 = (z1 - z2) * xperp + (z1 - m00) * yperp;
                
                /* Right point */
                z1 = s_mag[(s_row + 1) * (blockDim.x + 2) + s_col];
                z2 = s_mag[(s_row + 1) * (blockDim.x + 2) + (s_col + 1)];
                
                mag2 = (z1 - z2) * xperp + (z1 - m00) * yperp;
            }
            } else {
            if (gx >= -gy) {
                /* 101 */
                /* Left point */
                z1 = s_mag[s_row * (blockDim.x + 2) + (s_col - 1)];
                z2 = s_mag[(s_row + 1) * (blockDim.x + 2) + (s_col - 1)];
                
                mag1 = (m00 - z1) * xperp + (z1 - z2) * yperp;
                
                /* Right point */
                z1 = s_mag[s_row * (blockDim.x + 2) + (s_col + 1)];
                z2 = s_mag[(s_row - 1) * (blockDim.x + 2) + (s_col + 1)];
                
                mag2 = (m00 - z1) * xperp + (z1 - z2) * yperp;
            } else {
                /* 100 */
                /* Left point */
                z1 = s_mag[(s_row + 1) * (blockDim.x + 2) + s_col];
                z2 = s_mag[(s_row + 1) * (blockDim.x + 2) + (s_col - 1)];
                
                mag1 = (z1 - z2) * xperp + (m00 - z1) * yperp;
                
                /* Right point */
                z1 = s_mag[(s_row - 1) * (blockDim.x + 2) + s_col];
                z2 = s_mag[(s_row - 1) * (blockDim.x + 2) + (s_col + 1)];
                
                mag2 = (z1 - z2) * xperp + (m00 - z1) * yperp;
            }
            }
        } else {
            if (gy >= 0) {
            if (-gx >= gy) {
                /* 011 */
                /* Left point */
                z1 = s_mag[s_row * (blockDim.x + 2) + (s_col + 1)];
                z2 = s_mag[(s_row - 1) * (blockDim.x + 2) + (s_col + 1)];
                
                mag1 = (z1 - m00) * xperp + (z2 - z1) * yperp;
                
                /* Right point */
                z1 = s_mag[s_row * (blockDim.x + 2) + (s_col - 1)];
                z2 = s_mag[(s_row + 1) * (blockDim.x + 2) + (s_col - 1)];
                
                mag2 = (z1 - m00) * xperp + (z2 - z1) * yperp;
            } else {
                /* 010 */
                /* Left point */
                z1 = s_mag[(s_row - 1) * (blockDim.x + 2) + s_col];
                z2 = s_mag[(s_row - 1) * (blockDim.x + 2) + (s_col + 1)];
                
                mag1 = (z2 - z1) * xperp + (z1 - m00) * yperp;
                
                /* Right point */
                z1 = s_mag[(s_row + 1) * (blockDim.x + 2) + s_col];
                z2 = s_mag[(s_row + 1) * (blockDim.x + 2) + (s_col - 1)];
                
                mag2 = (z2 - z1) * xperp + (z1 - m00) * yperp;
            }
            } else {
            if (-gx > -gy) {
                /* 001 */
                /* Left point */
                z1 = s_mag[s_row * (blockDim.x + 2) + (s_col + 1)];
                z2 = s_mag[(s_row + 1) * (blockDim.x + 2) + (s_col + 1)];
                
                mag1 = (z1 - m00) * xperp + (z1 - z2) * yperp;
                
                /* Right point */
                z1 = s_mag[s_row * (blockDim.x + 2) + (s_col - 1)];
                z2 = s_mag[(s_row - 1) * (blockDim.x + 2) + (s_col - 1)];
                
                mag2 = (z1 - m00) * xperp + (z1 - z2) * yperp;
            } else {
                /* 000 */
                /* Left point */
                z1 = s_mag[(s_row + 1) * (blockDim.x + 2) + s_col];
                z2 = s_mag[(s_row + 1) * (blockDim.x + 2) + (s_col + 1)];
                
                mag1 = (z2 - z1) * xperp + (m00 - z1) * yperp;
                
                /* Right point */
                z1 = s_mag[(s_row - 1) * (blockDim.x + 2) + s_col];
                z2 = s_mag[(s_row - 1) * (blockDim.x + 2) + (s_col - 1)];
                
                mag2 = (z2 - z1) * xperp + (m00 - z1) * yperp;
            }
            }
        }
        
        // Less branching
        unsigned char edge_val = NOEDGE;
        if (mag1 <= 0.0f && mag2 <= 0.0f) {
            if (mag2 < 0.0f) {
                edge_val = POSSIBLE_EDGE;
            }
        }
        
        result[row * d_ncols + col] = edge_val;
    }
}

__global__ void init_borders_kernel(unsigned char* result) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < d_ncols) {
        // Top row
        result[col] = NOEDGE;
        // Bottom row
        result[(d_nrows-1) * d_ncols + col] = NOEDGE;
    }
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < d_nrows) {
        // Left column
        result[row * d_ncols] = NOEDGE;
        // Right column
        result[row * d_ncols + d_ncols - 1] = NOEDGE;
    }
}

int cuda_non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols, unsigned char *result) {

    short *d_mag, *d_gradx, *d_grady;
    unsigned char *d_result;

    cudaError_t cudaStatus;

    cudaEvent_t start, stop;
    float gpu_time = 0.0f;
    
    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }
    
    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaEventDestroy(start);
        return -1;
    }
    
    size_t size_short = nrows * ncols * sizeof(short);
    size_t size_uchar = nrows * ncols * sizeof(unsigned char);
    
    // Dimensions in constant memory
    cudaEventRecord(start);
    cudaStatus = cudaMemcpyToSymbol(d_nrows, &nrows, sizeof(int));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpyToSymbol d_nrows time: %.2f ms\n", gpu_time);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol d_nrows failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }
    
    cudaEventRecord(start);
    cudaStatus = cudaMemcpyToSymbol(d_ncols, &ncols, sizeof(int));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpyToSymbol d_ncols time: %.2f ms\n", gpu_time);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol d_ncols failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }
    
    cudaEventRecord(start);
    cudaStatus = cudaMalloc((void**)&d_mag, size_short);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc d_mag time: %.2f ms\n", gpu_time);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_mag failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }
    
    cudaEventRecord(start);
    cudaStatus = cudaMalloc((void**)&d_gradx, size_short);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc d_gradx time: %.2f ms\n", gpu_time);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_gradx failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_mag);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }
    
    cudaEventRecord(start);
    cudaStatus = cudaMalloc((void**)&d_grady, size_short);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc d_grady time: %.2f ms\n", gpu_time);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_grady failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_mag);
        cudaFree(d_gradx);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }
    
    cudaEventRecord(start);
    cudaStatus = cudaMalloc((void**)&d_result, size_uchar);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMalloc d_result time: %.2f ms\n", gpu_time);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_result failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_mag);
        cudaFree(d_gradx);
        cudaFree(d_grady);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }
    
    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(d_mag, mag, size_short, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy d_mag time: %.2f ms\n", gpu_time);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_mag failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_mag);
        cudaFree(d_gradx);
        cudaFree(d_grady);
        cudaFree(d_result);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }
    
    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(d_gradx, gradx, size_short, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy d_gradx time: %.2f ms\n", gpu_time);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_gradx failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_mag);
        cudaFree(d_gradx);
        cudaFree(d_grady);
        cudaFree(d_result);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }
    
    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(d_grady, grady, size_short, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy d_grady time: %.2f ms\n", gpu_time);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_grady failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_mag);
        cudaFree(d_gradx);
        cudaFree(d_grady);
        cudaFree(d_result);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }
    
    // Initialise with 0
    cudaEventRecord(start);
    cudaStatus = cudaMemset(d_result, 0, size_uchar);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemset d_result time: %.2f ms\n", gpu_time);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset d_result failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_mag);
        cudaFree(d_gradx);
        cudaFree(d_grady);
        cudaFree(d_result);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }
    
    // First kernel: initialize borders
    dim3 borderBlockSize(256, 1);
    dim3 borderGridSizeX((ncols + borderBlockSize.x - 1) / borderBlockSize.x, 1);
    dim3 borderGridSizeY(1, (nrows + borderBlockSize.y - 1) / borderBlockSize.y);
    
    cudaEventRecord(start);
    init_borders_kernel<<<borderGridSizeX, borderBlockSize>>>(d_result);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("init_borders_kernel X execution time: %.2f ms\n", gpu_time);
    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "init_borders_kernel X launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_mag);
        cudaFree(d_gradx);
        cudaFree(d_grady);
        cudaFree(d_result);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }
    
    cudaEventRecord(start);
    init_borders_kernel<<<borderGridSizeY, borderBlockSize>>>(d_result);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("init_borders_kernel Y execution time: %.2f ms\n", gpu_time);
    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "init_borders_kernel Y launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_mag);
        cudaFree(d_gradx);
        cudaFree(d_grady);
        cudaFree(d_result);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }
    
    // Second kernel: non-maximum suppression
    int blockWidth = 32;
    int blockHeight = 16;
    
    dim3 blockSize(blockWidth, blockHeight);
    dim3 gridSize((ncols + blockSize.x - 1) / blockSize.x, 
                  (nrows + blockSize.y - 1) / blockSize.y);
    
    size_t sharedMemSize = 3 * (blockWidth + 2) * (blockHeight + 2) * sizeof(short);
    
    cudaEventRecord(start);
    non_max_supp_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_mag, d_gradx, d_grady, d_result);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("non_max_supp_kernel execution time: %.2f ms\n", gpu_time);
    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "non_max_supp_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_mag);
        cudaFree(d_gradx);
        cudaFree(d_grady);
        cudaFree(d_result);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }
    
    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(result, d_result, size_uchar, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("cudaMemcpy result to host time: %.2f ms\n", gpu_time);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy result failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_mag);
        cudaFree(d_gradx);
        cudaFree(d_grady);
        cudaFree(d_result);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }
    
    // Cleanup
    cudaFree(d_mag);
    cudaFree(d_gradx);
    cudaFree(d_grady);
    cudaFree(d_result);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}