#ifndef CANNY_EDGE_H
#define CANNY_EDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "pgm_io.h"
#include "hysteresis.h"

typedef long long fixed;
#define fixeddot 16

#define VERBOSE 1
#define BOOSTBLURFACTOR 90.0

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

/**
 * @brief Create a Gaussian kernel.
 *
 * @param sigma Sigma for Gaussian kernel
 * @param kernel Output kernel array
 * @param windowsize Output window size
 */
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize);

/**
 * @brief Compute X and Y derivatives of an image.
 *
 * @param smoothedim Input smoothed image
 * @param rows Number of rows
 * @param cols Number of columns
 * @param delta_x Output X derivative
 * @param delta_y Output Y derivative
 */
void derivative_x_y(short int *smoothedim, int rows, int cols, short int **delta_x, short int **delta_y);

/**
 * @brief Compute gradient magnitude from X and Y derivatives.
 *
 * @param delta_x X derivative
 * @param delta_y Y derivative
 * @param rows Number of rows
 * @param cols Number of columns
 * @param magnitude Output magnitude image
 */
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols, short int **magnitude);

/**
 * @brief Apply hysteresis thresholding to detect edges.
 *
 * @param mag Magnitude image
 * @param nms Non-maximal suppression image
 * @param rows Number of rows
 * @param cols Number of columns
 * @param tlow Low threshold
 * @param thigh High threshold
 * @param edge Output edge image
 */
void apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols, float tlow, float thigh, unsigned char *edge);

/**
 * @brief Compute gradient direction in radians.
 *
 * @param delta_x X derivative
 * @param delta_y Y derivative
 * @param rows Number of rows
 * @param cols Number of columns
 * @param dir_radians Output direction image
 * @param xdirtag X direction tag (-1 or 1)
 * @param ydirtag Y direction tag (-1 or 1)
 */
void radian_direction(short int *delta_x, short int *delta_y, int rows, int cols, float **dir_radians, int xdirtag, int ydirtag);

/**
 * @brief Compute angle in radians from X and Y components.
 *
 * @param x X component
 * @param y Y component
 * @return double Angle in radians (0 to 2π)
 */
double angle_radians(double x, double y);

#ifdef __cplusplus
}
#endif

#endif // CANNY_EDGE_H