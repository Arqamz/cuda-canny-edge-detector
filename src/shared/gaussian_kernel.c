#include "gaussian_kernel.h"

/*******************************************************************************
 * PROCEDURE: make_gaussian_kernel
 * PURPOSE: Create a one dimensional gaussian kernel.
 * NAME: Mike Heath
 * DATE: 2/15/96
 *******************************************************************************/
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize)
{
   int i, center;
   float x, fx, sum = 0.0;

   *windowsize = 1 + 2 * ceil(2.5 * sigma);
   center = (*windowsize) / 2;

   if (VERBOSE)
      printf("-Kernel of %d elements created.\n", *windowsize);
   if ((*kernel = (float *)malloc((*windowsize) * sizeof(float))) == NULL)
   {
      fprintf(stderr, "Error callocing the gaussian kernel array.\n");
      exit(1);
   }

   for (i = 0; i < (*windowsize); i++)
   {
      x = (float)(i - center);
      fx = pow(2.71828, -0.5 * x * x / (sigma * sigma)) / (sigma * sqrt(6.2831853));
      (*kernel)[i] = fx;
      sum += fx;
   }

   for (i = 0; i < (*windowsize); i++)
      (*kernel)[i] /= sum;

   // if (VERBOSE)
   // {
   //    printf("The filter coefficients are:\n");
   //    for (i = 0; i < (*windowsize); i++)
   //       printf("kernel[%d] = %f\n", i, (*kernel)[i]);
   // }
}
