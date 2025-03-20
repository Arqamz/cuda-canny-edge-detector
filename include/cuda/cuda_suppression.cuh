#ifndef CUDA_SUPPRESSION_CUH
#define cuda_suppCUDA_SUPPRESSION_CUH

#include <stdio.h>

#include <cuda_runtime.h>

__global__ void init_borders_kernel(unsigned char* result);
__global__ void non_max_supp_kernel(const short *mag, const short *gradx, const short *grady, unsigned char *result);

int cuda_non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols, unsigned char *result);

#endif