#ifndef CUDA_CANNY_EDGE_CUH
#define CUDA_CANNY_EDGE_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include "canny_edge.h"

// CUDA kernel for Gaussian blur in X direction
__global__ void gaussian_smooth_x_kernel(const unsigned char *d_image, float *d_temp, int rows, int cols, int center, int windowsize);

// CUDA kernel for Gaussian blur in Y direction
__global__ void gaussian_smooth_y_kernel(const float *d_temp, short int *d_smoothed, int rows, int cols, int center, int windowsize);

// Create a Gaussian kernel with specified sigma
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize);

// Apply Gaussian smoothing using CUDA
void cuda_gaussian_smooth(unsigned char *image, int rows, int cols, float sigma, short int **smoothedim);

// Main function to perform Canny edge detection using CUDA
void cuda_canny(unsigned char *image, int rows, int cols, float sigma, float tlow, float thigh, unsigned char **edge, char *fname);

#endif  // CUDA_CANNY_EDGE_CUH