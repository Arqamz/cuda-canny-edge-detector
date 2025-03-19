#include "magnitude.h"

/*******************************************************************************
 * PROCEDURE: magnitude_x_y
 * PURPOSE: Compute the magnitude of the gradient. This is the square root of
 * the sum of the squared derivative values.
 * NAME: Mike Heath
 * DATE: 2/15/96
 *******************************************************************************/
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols, short int **magnitude)
{
   int r, c, pos, sq1, sq2;

   /****************************************************************************
    * Allocate an image to store the magnitude of the gradient.
    ****************************************************************************/
   if ((*magnitude = (short *)malloc(rows * cols * sizeof(short))) == NULL)
   {
      fprintf(stderr, "Error allocating the magnitude image.\n");
      exit(1);
   }

   for (r = 0, pos = 0; r < rows; r++)
   {
      for (c = 0; c < cols; c++, pos++)
      {
         sq1 = (int)delta_x[pos] * (int)delta_x[pos];
         sq2 = (int)delta_y[pos] * (int)delta_y[pos];
         (*magnitude)[pos] = (short)(0.5 + sqrt((float)sq1 + (float)sq2));
      }
   }
}
