#ifndef CUDA_MAGNITUDE_CUH
#define CUDA_MAGNITUDE_CUH

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

__global__ void magnitude_kernel(short int *delta_x, short int *delta_y, short int *magnitude, int rows, int cols);

void cuda_magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols, short int **magnitude);

#endif