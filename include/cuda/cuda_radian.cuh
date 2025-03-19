#ifndef CUDA_RADIAN_DIRECTION_CUH
#define CUDA_RADIAN_DIRECTION_CUH

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <math_constants.h>

#include <cuda_runtime.h>

__constant__ int c_xdirtag;
__constant__ int c_ydirtag;

__global__ void compute_radian_direction_kernel(const short int* delta_x, const short int* delta_y, float* dir_radians, int rows, int cols);

void cuda_radian_direction(short int *delta_x, short int *delta_y, int rows, int cols, float **dir_radians, int xdirtag, int ydirtag);

#endif