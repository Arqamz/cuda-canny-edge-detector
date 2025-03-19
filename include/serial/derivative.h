#ifndef DERIVATIVE_H
#define DERIVATIVE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

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


#ifdef __cplusplus
}
#endif

#endif // DERIVATIVE_H