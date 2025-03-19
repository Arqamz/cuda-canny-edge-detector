#ifndef CUDA_GAUSSIAN_SMOOTH_CUH
#define CUDA_GAUSSIAN_SMOOTH_CUH

#define MAX_KERNEL_SIZE 4096

#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "canny_edge.h"

// Constants for the GPU constant memory
__constant__ struct {
    float kernel[MAX_KERNEL_SIZE];
    float sum;
} d_gaussian_constants;

// CUDA kernel for Gaussian blur in X direction
__global__ void gaussian_smooth_x_kernel(const unsigned char *d_image, float *d_temp, int rows, int cols, int kernel_radius);

// CUDA kernel for Gaussian blur in Y direction
__global__ void gaussian_smooth_y_kernel(const float *d_temp, short int *d_smoothed, int rows, int cols, int kernel_radius);

// Apply Gaussian smoothing using CUDA
void cuda_gaussian_smooth(unsigned char *image, int rows, int cols, float sigma, short int **smoothedim);

#endif  // CUDA_GAUSSIAN_SMOOTH_CUH