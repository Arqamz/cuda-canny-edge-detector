#include "derivative.h"

/*******************************************************************************
 * PROCEDURE: derivative_x_y
 * PURPOSE: Compute the first derivative of the image in both the x any y
 * directions. The differential filters that are used are:
 *
 *                                          -1
 *         dx =  -1 0 +1     and       dy =  0
 *                                          +1
 *
 * NAME: Mike Heath
 * DATE: 2/15/96
 *******************************************************************************/
void derivative_x_y(short int *smoothedim, int rows, int cols, short int **delta_x, short int **delta_y)
{
   int r, c, pos;
   /****************************************************************************
    * Allocate images to store the derivatives.
    ****************************************************************************/
   if (((*delta_x) = (short *)malloc(rows * cols * sizeof(short))) == NULL)
   {
      fprintf(stderr, "Error allocating the delta_x image.\n");
      exit(1);
   }
   if (((*delta_y) = (short *)malloc(rows * cols * sizeof(short))) == NULL)
   {
      fprintf(stderr, "Error allocating the delta_y image.\n");
      exit(1);
   }

   for (r = 0; r < rows; r++)
   {
      pos = r * cols;
      (*delta_x)[pos] = smoothedim[pos + 1] - smoothedim[pos];
      pos++;
      for (c = 1; c < (cols - 1); c++, pos++)
      {
         (*delta_x)[pos] = smoothedim[pos + 1] - smoothedim[pos - 1];
      }
      (*delta_x)[pos] = smoothedim[pos] - smoothedim[pos - 1];
   }

   for (c = 0; c < cols; c++)
   {
      pos = c;
      (*delta_y)[pos] = smoothedim[pos + cols] - smoothedim[pos];
      pos += cols;
      for (r = 1; r < (rows - 1); r++, pos += cols)
      {
         (*delta_y)[pos] = smoothedim[pos + cols] - smoothedim[pos - cols];
      }
      (*delta_y)[pos] = smoothedim[pos] - smoothedim[pos - cols];
   }
}
