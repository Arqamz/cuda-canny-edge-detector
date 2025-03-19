#ifndef GAUSSIAN_SMOOTH_H
#define GAUSSIAN_SMOOTH_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

#include "gaussian_kernel.h"

#define VERBOSE 1
#define BOOSTBLURFACTOR 90.0

/**
 * @brief Apply Gaussian smoothing to an image.
 *
 * @param image Input image data
 * @param rows Number of rows
 * @param cols Number of columns
 * @param sigma Sigma for Gaussian filter
 * @param smoothedim Output smoothed image
 */
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma, short int **smoothedim);

#ifdef __cplusplus
}
#endif

#endif // GAUSSIAN_SMOOTH_H