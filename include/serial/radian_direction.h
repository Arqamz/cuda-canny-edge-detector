#ifndef RADIAN_DIRECTION_H
#define RADIAN_DIRECTION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

#endif // RADIAN_DIRECTION_H