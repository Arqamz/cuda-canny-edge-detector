#ifndef CANNY_EDGE_H
#define CANNY_EDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

#include "timer.h"
#include "pgm_io.h"
#include "gaussian_smooth.h"
#include "derivative.h"
#include "radian_direction.h"
#include "magnitude.h"
#include "hysteresis.h"

#define VERBOSE 1

/**
 * @brief Perform Canny edge detection on an image.
 *
 * @param image Input image data
 * @param rows Number of rows
 * @param cols Number of columns
 * @param sigma Sigma for Gaussian smoothing
 * @param tlow Low threshold for hysteresis
 * @param thigh High threshold for hysteresis
 * @param edge Output edge image
 * @param fname Optional filename for gradient direction output
 */
void canny(unsigned char *image, int rows, int cols, float sigma, float tlow, float thigh, unsigned char **edge, char *fname);

#ifdef __cplusplus
}
#endif

#endif // CANNY_EDGE_H