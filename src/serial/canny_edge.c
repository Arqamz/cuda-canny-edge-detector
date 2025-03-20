/*******************************************************************************
 * --------------------------------------------
 *(c) 2001 University of South Florida, Tampa
 * Use, or copying without permission prohibited.
 * PERMISSION TO USE
 * In transmitting this software, permission to use for research and
 * educational purposes is hereby granted.  This software may be copied for
 * archival and backup purposes only.  This software may not be transmitted
 * to a third party without prior permission of the copyright holder. This
 * permission may be granted only by Mike Heath or Prof. Sudeep Sarkar of
 * University of South Florida (sarkar@csee.usf.edu). Acknowledgment as
 * appropriate is respectfully requested.
 *
 *  Heath, M., Sarkar, S., Sanocki, T., and Bowyer, K. Comparison of edge
 *    detectors: a methodology and initial study, Computer Vision and Image
 *    Understanding 69 (1), 38-54, January 1998.
 *  Heath, M., Sarkar, S., Sanocki, T. and Bowyer, K.W. A Robust Visual
 *    Method for Assessing the Relative Performance of Edge Detection
 *    Algorithms, IEEE Transactions on Pattern Analysis and Machine
 *    Intelligence 19 (12),  1338-1359, December 1997.
 *  ------------------------------------------------------
 *
 * PROGRAM: canny_edge
 * PURPOSE: This program implements a "Canny" edge detector. The processing
 * steps are as follows:
 *
 *   1) Convolve the image with a separable gaussian filter.
 *   2) Take the dx and dy the first derivatives using [-1,0,1] and [1,0,-1]'.
 *   3) Compute the magnitude: sqrt(dx*dx+dy*dy).
 *   4) Perform non-maximal suppression.
 *   5) Perform hysteresis.
 *
 * The user must input three parameters. These are as follows:
 *
 *   sigma = The standard deviation of the gaussian smoothing filter.
 *   tlow  = Specifies the low value to use in hysteresis. This is a
 *           fraction (0-1) of the computed high threshold edge strength value.
 *   thigh = Specifies the high value to use in hysteresis. This fraction (0-1)
 *           specifies the percentage point in a histogram of the gradient of
 *           the magnitude. Magnitude values of zero are not counted in the
 *           histogram.
 *
 * NAME: Mike Heath
 *       Computer Vision Laboratory
 *       University of South Floeida
 *       heath@csee.usf.edu
 *
 * DATE: 2/15/96
 *
 * Modified: 5/17/96 - To write out a floating point RAW headerless file of
 *                     the edge gradient "up the edge" where the angle is
 *                     defined in radians counterclockwise from the x direction.
 *                     (Mike Heath)
 *******************************************************************************/

#include "canny_edge.h"

/*******************************************************************************
 * PROCEDURE: canny
 * PURPOSE: To perform canny edge detection.
 * NAME: Mike Heath
 * DATE: 2/15/96
 *******************************************************************************/
void canny(unsigned char *image, int rows, int cols, float sigma, float tlow, float thigh, unsigned char **edge, char *fname)
{
   FILE *fpdir = NULL;    /* File to write the gradient image to.     */
   unsigned char *nms;    /* Points that are local maximal magnitude. */
   short int *smoothedim, /* The image after gaussian smoothing.      */
       *delta_x,          /* The first devivative image, x-direction. */
       *delta_y,          /* The first derivative image, y-direction. */
       *magnitude;        /* The magnitude of the gadient image.      */
   float *dir_radians = NULL; /* Gradient direction image.                */

   // Variables for timing
   double start_time, end_time, step_time;
   double total_time = 0.0;

   /****************************************************************************
    * Perform gaussian smoothing on the image using the input standard
    * deviation.
    ****************************************************************************/
   if (VERBOSE)
      printf("Smoothing the image using a gaussian kernel.\n");
   
   start_time = get_time_ms();
   gaussian_smooth(image, rows, cols, sigma, &smoothedim);
   end_time = get_time_ms();
   step_time = end_time - start_time;
   total_time += step_time;
   printf("==================================\n");
   printf("Gaussian smoothing time: %.2f ms\n", step_time);
   printf("==================================\n\n");

   /****************************************************************************
    * Compute the first derivative in the x and y directions.
    ****************************************************************************/
   if (VERBOSE)
      printf("Computing the X and Y first derivatives.\n");
   
   start_time = get_time_ms();
   derivative_x_y(smoothedim, rows, cols, &delta_x, &delta_y);
   end_time = get_time_ms();
   step_time = end_time - start_time;
   total_time += step_time;
   printf("==================================\n");
   printf("X and Y derivatives computation time: %.2f ms\n", step_time);
   printf("==================================\n\n");

   /****************************************************************************
    * This option to write out the direction of the edge gradient was added
    * to make the information available for computing an edge quality figure
    * of merit.
    ****************************************************************************/
   if (fname != NULL)
   {
      start_time = get_time_ms();
      /*************************************************************************
       * Compute the direction up the gradient, in radians that are
       * specified counteclockwise from the positive x-axis.
       *************************************************************************/
      radian_direction(delta_x, delta_y, rows, cols, &dir_radians, -1, -1);

      /*************************************************************************
       * Write the gradient direction image out to a file.
       *************************************************************************/
      if ((fpdir = fopen(fname, "wb")) == NULL)
      {
         fprintf(stderr, "Error opening the file %s for writing.\n", fname);
         exit(1);
      }
      fwrite(dir_radians, sizeof(float), rows * cols, fpdir);
      fclose(fpdir);
      end_time = get_time_ms();
      step_time = end_time - start_time;
      total_time += step_time;
      printf("==================================\n");
      printf("Direction calculation time: %.2f ms\n", step_time);
      printf("==================================\n\n");
      
      free(dir_radians);
   }

   /****************************************************************************
    * Compute the magnitude of the gradient.
    ****************************************************************************/
   if (VERBOSE)
      printf("Computing the magnitude of the gradient.\n");
   
   start_time = get_time_ms();
   magnitude_x_y(delta_x, delta_y, rows, cols, &magnitude);
   end_time = get_time_ms();
   step_time = end_time - start_time;
   total_time += step_time;
   printf("==================================\n");
   printf("Gradient magnitude computation time: %.2f ms\n", step_time);
   printf("==================================\n\n");

   /****************************************************************************
    * Perform non-maximal suppression.
    ****************************************************************************/
   if (VERBOSE)
      printf("Doing the non-maximal suppression.\n");
   
   start_time = get_time_ms();
   if ((nms = (unsigned char *)malloc(rows * cols * sizeof(unsigned char))) == NULL)
   {
      fprintf(stderr, "Error allocating the nms image.\n");
      exit(1);
   }
   non_max_supp(magnitude, delta_x, delta_y, rows, cols, nms);
   end_time = get_time_ms();
   step_time = end_time - start_time;
   total_time += step_time;
   printf("==================================\n");
   printf("Non-maximal suppression time: %.2f ms\n", step_time);
   printf("==================================\n\n");

   /****************************************************************************
    * Use hysteresis to mark the edge pixels.
    ****************************************************************************/
   if (VERBOSE)
      printf("Doing hysteresis thresholding.\n");
   
   start_time = get_time_ms();
   if ((*edge = (unsigned char *)malloc(rows * cols * sizeof(unsigned char))) == NULL)
   {
      fprintf(stderr, "Error allocating the edge image.\n");
      exit(1);
   }
   apply_hysteresis(magnitude, nms, rows, cols, tlow, thigh, *edge);
   end_time = get_time_ms();
   step_time = end_time - start_time;
   total_time += step_time;
   printf("==================================\n");
   printf("Hysteresis thresholding time: %.2f ms\n", step_time);
   printf("==================================\n\n");

   // Print total time
   printf("====================================================================\n");
   printf("Total Canny edge detection time: %.2f ms\n", total_time);

   /****************************************************************************
    * Free all of the memory that we allocated except for the edge image that
    * is still being used to store out result.
    ****************************************************************************/
   free(smoothedim);
   free(delta_x);
   free(delta_y);
   free(magnitude);
   free(nms);
}