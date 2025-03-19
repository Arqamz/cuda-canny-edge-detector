#ifndef CUDA_CANNY_EDGE_CUH
#define CUDA_CANNY_EDGE_CUH

#define VERBOSE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "canny_edge.h"

#include "cuda_gaussian_smooth.cuh"
#include "cuda_derivative.cuh"

// Main function to perform Canny edge detection using CUDA
void cuda_canny(unsigned char *image, int rows, int cols, float sigma, float tlow, float thigh, unsigned char **edge, char *fname);

#endif  // CUDA_CANNY_EDGE_CUH