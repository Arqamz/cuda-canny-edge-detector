#ifndef MAGNITUDE_H
#define MAGNITUDE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

#ifdef __cplusplus
}
#endif

#endif // MAGNITUDE_H