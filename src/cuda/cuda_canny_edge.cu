#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_canny_edge.cuh"

#define VERBOSE 1
#define MAX_KERNEL_SIZE 128

// Constants for the GPU constant memory
__constant__ float d_kernel[MAX_KERNEL_SIZE];

// Kernel for horizontal Gaussian smoothing
__global__ void gaussian_smooth_x_kernel(const unsigned char *d_image, float *d_temp, 
                                        int rows, int cols, int center, int windowsize)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(r < rows && c < cols)
    {
        float dot = 0.0f, sum = 0.0f;
        for (int cc = -center; cc <= center; cc++) {
            int col = c + cc;
            if(col >= 0 && col < cols) {
                dot += d_image[r * cols + col] * d_kernel[center + cc];
                sum += d_kernel[center + cc];
            }
        }
        d_temp[r * cols + c] = dot / sum;
    }
}

// Kernel for vertical Gaussian smoothing
__global__ void gaussian_smooth_y_kernel(const float *d_temp, short int *d_smoothed, 
                                        int rows, int cols, int center, int windowsize)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(r < rows && c < cols)
    {
        float dot = 0.0f, sum = 0.0f;
        for (int rr = -center; rr <= center; rr++) {
            int row = r + rr;
            if(row >= 0 && row < rows) {
                dot += d_temp[row * cols + c] * d_kernel[center + rr];
                sum += d_kernel[center + rr];
            }
        }
        // Apply the BOOSTBLURFACTOR to match the serial implementation
        d_smoothed[r * cols + c] = (short int)(dot * 90.0f / sum + 0.5f);
    }
}

// Function to create Gaussian kernel and call GPU kernels
void cuda_gaussian_smooth(unsigned char *image, int rows, int cols, float sigma, short int **smoothedim)
{
    int windowsize, center;
    float *kernel;
    
    // Create the Gaussian kernel
    make_gaussian_kernel(sigma, &kernel, &windowsize);
    center = windowsize / 2;
    
    // Copy kernel to constant memory
    cudaMemcpyToSymbol(d_kernel, kernel, windowsize * sizeof(float));
    free(kernel);  // Free CPU kernel memory after copying to GPU

    // Calculate sizes for memory allocation
    size_t imageSize = rows * cols * sizeof(unsigned char);
    size_t tempSize = rows * cols * sizeof(float);
    size_t smoothSize = rows * cols * sizeof(short int);

    // Allocate device memory
    unsigned char *d_image;
    float *d_temp;
    short int *d_smoothed;
    cudaMalloc((void **)&d_image, imageSize);
    cudaMalloc((void **)&d_temp, tempSize);
    cudaMalloc((void **)&d_smoothed, smoothSize);
    
    // Copy input image to device
    cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
    
    // Launch kernels
    gaussian_smooth_x_kernel<<<gridDim, blockDim>>>(d_image, d_temp, rows, cols, center, windowsize);
    cudaDeviceSynchronize();
    
    gaussian_smooth_y_kernel<<<gridDim, blockDim>>>(d_temp, d_smoothed, rows, cols, center, windowsize);
    cudaDeviceSynchronize();

    // Allocate host memory for result and copy from device
    *smoothedim = (short int *)malloc(smoothSize);
    if(*smoothedim == NULL) {
        fprintf(stderr, "Error allocating the smoothedim array.\n");
        exit(1);
    }
    
    cudaMemcpy(*smoothedim, d_smoothed, smoothSize, cudaMemcpyDeviceToHost);
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // Free device memory
    cudaFree(d_image);
    cudaFree(d_temp);
    cudaFree(d_smoothed);
}

// GPU-accelerated Canny edge detection (only Gaussian smoothing on GPU)
void cuda_canny(unsigned char *image, int rows, int cols, float sigma, 
                float tlow, float thigh, unsigned char **edge, char *fname)
{
    FILE *fpdir = NULL;    /* File to write the gradient image to.     */
    unsigned char *nms;    /* Points that are local maximal magnitude. */
    short int *smoothedim, /* The image after gaussian smoothing.      */
       *delta_x,          /* The first devivative image, x-direction. */
       *delta_y,          /* The first derivative image, y-direction. */
       *magnitude;        /* The magnitude of the gadient image.      */
    float *dir_radians = NULL; /* Gradient direction image.                */

    /****************************************************************************
     * Perform gaussian smoothing on the image using GPU.
     ****************************************************************************/
    if (VERBOSE)
        printf("Smoothing the image using a gaussian kernel on GPU.\n");
    cuda_gaussian_smooth(image, rows, cols, sigma, &smoothedim);

    /****************************************************************************
     * Compute the first derivative in the x and y directions (CPU).
     ****************************************************************************/
    if (VERBOSE)
        printf("Computing the X and Y first derivatives.\n");
    derivative_x_y(smoothedim, rows, cols, &delta_x, &delta_y);

    /****************************************************************************
     * Direction calculation for edge quality figure of merit (CPU).
     ****************************************************************************/
    if (fname != NULL)
    {
        radian_direction(delta_x, delta_y, rows, cols, &dir_radians, -1, -1);

        if ((fpdir = fopen(fname, "wb")) == NULL)
        {
            fprintf(stderr, "Error opening the file %s for writing.\n", fname);
            exit(1);
        }
        fwrite(dir_radians, sizeof(float), rows * cols, fpdir);
        fclose(fpdir);
        free(dir_radians);
    }

    /****************************************************************************
     * Compute the magnitude of the gradient (CPU).
     ****************************************************************************/
    if (VERBOSE)
        printf("Computing the magnitude of the gradient.\n");
    magnitude_x_y(delta_x, delta_y, rows, cols, &magnitude);

    /****************************************************************************
     * Perform non-maximal suppression (CPU).
     ****************************************************************************/
    if (VERBOSE)
        printf("Doing the non-maximal suppression.\n");
    if ((nms = (unsigned char *)malloc(rows * cols * sizeof(unsigned char))) == NULL)
    {
        fprintf(stderr, "Error allocating the nms image.\n");
        exit(1);
    }
    non_max_supp(magnitude, delta_x, delta_y, rows, cols, nms);

    /****************************************************************************
     * Use hysteresis to mark the edge pixels (CPU).
     ****************************************************************************/
    if (VERBOSE)
        printf("Doing hysteresis thresholding.\n");
    if ((*edge = (unsigned char *)malloc(rows * cols * sizeof(unsigned char))) == NULL)
    {
        fprintf(stderr, "Error allocating the edge image.\n");
        exit(1);
    }
    apply_hysteresis(magnitude, nms, rows, cols, tlow, thigh, *edge);

    /****************************************************************************
     * Free allocated memory.
     ****************************************************************************/
    free(smoothedim);
    free(delta_x);
    free(delta_y);
    free(magnitude);
    free(nms);
}