#ifndef CUDA_DERIVATIVE_CUH
#define CUDA_DERIVATIVE_CUH

#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_WIDTH 32
#define TILE_HEIGHT 32
#define MAX_BLOCK_SIZE 32

__global__ void compute_delta_x_kernel(const short int *smoothedim, short int *delta_x, int rows, int cols);

__global__ void compute_delta_y_kernel(const short int *smoothedim, short int *delta_y, int rows, int cols);

void cuda_derivative_x_y(short int *smoothedim, int rows, int cols, short int **delta_x, short int **delta_y);

#endif  // CUDA_DERIVATIVE_CUH