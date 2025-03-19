#ifndef GAUSSIAN_KERNEL_H
#define GAUSSIAN_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VERBOSE 1

/**
 * @brief Create a Gaussian kernel.
 *
 * @param sigma Sigma for Gaussian kernel
 * @param kernel Output kernel array
 * @param windowsize Output window size
 */
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize);

#ifdef __cplusplus
}
#endif

#endif /* GAUSSIAN_KERNEL_H */