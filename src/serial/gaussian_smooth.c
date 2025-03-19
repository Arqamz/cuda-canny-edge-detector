#include "gaussian_smooth.h"

/*******************************************************************************
 * PROCEDURE: gaussian_smooth
 * PURPOSE: Blur an image with a gaussian filter.
 * NAME: Mike Heath
 * DATE: 2/15/96
 *******************************************************************************/
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma, short int **smoothedim)
{
   int r, c, rr, cc, /* Counter variables. */
       windowsize,   /* Dimension of the gaussian kernel. */
       center;       /* Half of the windowsize. */
   float *tempim,    /* Buffer for separable filter gaussian smoothing. */
       *kernel,      /* A one dimensional gaussian kernel. */
       dot,          /* Dot product summing variable. */
       sum;          /* Sum of the kernel weights variable. */

   /****************************************************************************
    * Create a 1-dimensional gaussian smoothing kernel.
    ****************************************************************************/
   if (VERBOSE)
      printf("-Computing the gaussian smoothing kernel.\n");
   make_gaussian_kernel(sigma, &kernel, &windowsize);
   center = windowsize / 2;

   /****************************************************************************
    * Allocate a temporary buffer image and the smoothed image.
    ****************************************************************************/
   if ((tempim = (float *)malloc(rows * cols * sizeof(float))) == NULL)
   {
      fprintf(stderr, "Error allocating the buffer image.\n");
      exit(1);
   }
   if (((*smoothedim) = (short int *)malloc(rows * cols * sizeof(short int))) == NULL)
   {
      fprintf(stderr, "Error allocating the smoothed image.\n");
      exit(1);
   }

   /****************************************************************************
    * Blur in the x - direction.
    ****************************************************************************/
   if (VERBOSE)
      printf("-Bluring the image in the X-direction.\n");
   for (r = 0; r < rows; r++)
   {
      for (c = 0; c < cols; c++)
      {
         dot = 0.0;
         sum = 0.0;
         for (cc = (-center); cc <= center; cc++)
         {
            if (((c + cc) >= 0) && ((c + cc) < cols))
            {
               dot += (float)image[r * cols + (c + cc)] * kernel[center + cc];
               sum += kernel[center + cc];
            }
         }
         tempim[r * cols + c] = dot / sum;
      }
   }

   /****************************************************************************
    * Blur in the y - direction.
    ****************************************************************************/
   if (VERBOSE)
      printf("-Bluring the image in the Y-direction.\n");
   for (c = 0; c < cols; c++)
   {
      for (r = 0; r < rows; r++)
      {
         sum = 0.0;
         dot = 0.0;
         for (rr = (-center); rr <= center; rr++)
         {
            if (((r + rr) >= 0) && ((r + rr) < rows))
            {
               dot += tempim[(r + rr) * cols + c] * kernel[center + rr];
               sum += kernel[center + rr];
            }
         }
         (*smoothedim)[r * cols + c] = (short int)(dot * BOOSTBLURFACTOR / sum + 0.5);
      }
   }

   free(tempim);
   free(kernel);
}