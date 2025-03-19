#include "cuda_canny_edge.cuh"

// GPU-accelerated Canny edge detection (currently only Gaussian smoothing on GPU)
void cuda_canny(unsigned char *image, int rows, int cols, float sigma,
                float tlow, float thigh, unsigned char **edge, char *fname)
{
    FILE *fpdir = NULL;        /* File to write the gradient image to.     */
    unsigned char *nms;        /* Points that are local maximal magnitude. */
    short int *smoothedim,     /* The image after gaussian smoothing.      */
        *delta_x,              /* The first devivative image, x-direction. */
        *delta_y,              /* The first derivative image, y-direction. */
        *magnitude;            /* The magnitude of the gadient image.      */
    float *dir_radians = NULL; /* Gradient direction image.                */

    // Variables for timing
    double start_time, end_time, step_time;
    double total_time = 0.0;

    /****************************************************************************
     * Perform gaussian smoothing on the image using GPU.
     ****************************************************************************/
    if (VERBOSE)
        printf("Smoothing the image using a gaussian kernel on GPU.\n");

    start_time = get_time_ms();
    cuda_gaussian_smooth(image, rows, cols, sigma, &smoothedim);
    end_time = get_time_ms();
    step_time = end_time - start_time;
    total_time += step_time;
    printf("Gaussian smoothing time: %.2f ms\n", step_time);

    /****************************************************************************
     * Compute the first derivative in the x and y directions (CPU).
     ****************************************************************************/
    if (VERBOSE)
        printf("Computing the X and Y first derivatives.\n");

    start_time = get_time_ms();
    derivative_x_y(smoothedim, rows, cols, &delta_x, &delta_y);
    end_time = get_time_ms();
    step_time = end_time - start_time;
    total_time += step_time;
    printf("X and Y derivatives computation time: %.2f ms\n", step_time);

    /****************************************************************************
     * Direction calculation for edge quality figure of merit (CPU).
     ****************************************************************************/
    if (fname != NULL)
    {
        start_time = get_time_ms();
        radian_direction(delta_x, delta_y, rows, cols, &dir_radians, -1, -1);

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
        printf("Direction calculation time: %.2f ms\n", step_time);

        free(dir_radians);
    }

    /****************************************************************************
     * Compute the magnitude of the gradient (CPU).
     ****************************************************************************/
    if (VERBOSE)
        printf("Computing the magnitude of the gradient.\n");

    start_time = get_time_ms();
    magnitude_x_y(delta_x, delta_y, rows, cols, &magnitude);
    end_time = get_time_ms();
    step_time = end_time - start_time;
    total_time += step_time;
    printf("Gradient magnitude computation time: %.2f ms\n", step_time);

    /****************************************************************************
     * Perform non-maximal suppression (CPU).
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
    printf("Non-maximal suppression time: %.2f ms\n", step_time);

    /****************************************************************************
     * Use hysteresis to mark the edge pixels (CPU).
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
    printf("Hysteresis thresholding time: %.2f ms\n", step_time);

    // Print total time
    printf("Total Canny edge detection time: %.2f ms\n", total_time);

    /****************************************************************************
     * Free allocated memory.
     ****************************************************************************/
    free(smoothedim);
    free(delta_x);
    free(delta_y);
    free(magnitude);
    free(nms);
}
