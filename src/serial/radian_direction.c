#include "radian_direction.h"

/*******************************************************************************
 * Procedure: radian_direction
 * Purpose: To compute a direction of the gradient image from component dx and
 * dy images. Because not all derriviatives are computed in the same way, this
 * code allows for dx or dy to have been calculated in different ways.
 *
 * FOR X:  xdirtag = -1  for  [-1 0  1]
 *         xdirtag =  1  for  [ 1 0 -1]
 *
 * FOR Y:  ydirtag = -1  for  [-1 0  1]'
 *         ydirtag =  1  for  [ 1 0 -1]'
 *
 * The resulting angle is in radians measured counterclockwise from the
 * xdirection. The angle points "up the gradient".
 *******************************************************************************/
void radian_direction(short int *delta_x, short int *delta_y, int rows, int cols, float **dir_radians, int xdirtag, int ydirtag)
{
   int r, c, pos;
   float *dirim = NULL;
   double dx, dy;

   /****************************************************************************
    * Allocate an image to store the direction of the gradient.
    ****************************************************************************/
   if ((dirim = (float *)malloc(rows * cols * sizeof(float))) == NULL)
   {
        fprintf(stderr, "Error allocating the gradient direction image.\n");
        exit(1);
   }
   *dir_radians = dirim;

   for (r = 0, pos = 0; r < rows; r++)
   {
        for (c = 0; c < cols; c++, pos++)
        {
            dx = (double)delta_x[pos];
            dy = (double)delta_y[pos];

            if (xdirtag == 1)
                dx = -dx;
            if (ydirtag == -1)
                dy = -dy;

            dirim[pos] = (float)angle_radians(dx, dy);
        }
   }
}

/*******************************************************************************
 * FUNCTION: angle_radians
 * PURPOSE: This procedure computes the angle of a vector with components x and
 * y. It returns this angle in radians with the answer being in the range
 * 0 <= angle <2*PI.
 *******************************************************************************/
double angle_radians(double x, double y)
{
   double xu, yu, ang;

   xu = fabs(x);
   yu = fabs(y);

   if ((xu == 0) && (yu == 0))
      return (0);

   ang = atan(yu / xu);

   if (x >= 0)
   {
        if (y >= 0)
            return (ang);
        else
            return (2 * M_PI - ang);
   }
   else
   {
        if (y >= 0)
            return (M_PI - ang);
        else
            return (M_PI + ang);
   }
}