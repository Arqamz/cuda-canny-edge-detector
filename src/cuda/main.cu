#include <stdio.h>

#include "cuda_canny_edge.cuh"

int main(int argc, char *argv[])
{
    char *infilename = NULL;   /* Name of the input image */
    char *dirfilename = NULL;  /* Name of the output gradient direction image */
    char outfilename[128];     /* Name of the output "edge" image */
    char composedfname[128];   /* Name of the output "direction" image */
    unsigned char *image;      /* The input image */
    unsigned char *edge;       /* The output edge image */
    int rows, cols;            /* Image dimensions */
    float sigma,               /* Gaussian kernel standard deviation */
          tlow,                /* Hysteresis low threshold fraction */
          thigh;               /* Hysteresis high threshold fraction */
    char basename[128];        /* Base filename without directory prefix */

    /***************************************************************************
     * Get the command line arguments.
     ***************************************************************************/
    if (argc < 5) {
        fprintf(stderr,"\n<USAGE> %s image sigma tlow thigh [writedirim]\n",argv[0]);
        fprintf(stderr,"\n      image:      An image to process. Must be in ");
        fprintf(stderr,"PGM format.\n");
        fprintf(stderr,"      sigma:      Standard deviation of the gaussian");
        fprintf(stderr," blur kernel.\n");
        fprintf(stderr,"      tlow:       Fraction (0.0-1.0) of the high ");
        fprintf(stderr,"edge strength threshold.\n");
        fprintf(stderr,"      thigh:      Fraction (0.0-1.0) of the distribution");
        fprintf(stderr," of non-zero edge\n                  strengths for ");
        fprintf(stderr,"hysteresis. The fraction is used to compute\n");
        fprintf(stderr,"                  the high edge strength threshold.\n");
        fprintf(stderr,"      writedirim: Optional argument to output ");
        fprintf(stderr,"a floating point");
        fprintf(stderr," direction image.\n\n");
        exit(1);
    }

    infilename = argv[1];
    sigma = atof(argv[2]);
    tlow  = atof(argv[3]);
    thigh = atof(argv[4]);

    /* Extract basename from infilename by removing "input/" prefix if present */
    if (strncmp(infilename, "input/", 6) == 0)
        strcpy(basename, infilename + 6);
    else
        strcpy(basename, infilename);

    if (argc == 6)
        dirfilename = infilename;
    else
        dirfilename = NULL;

    /****************************************************************************
    * Read in the image. This read function allocates memory for the image.
    ****************************************************************************/
    if (VERBOSE) printf("Reading the image %s.\n", infilename);
    if (read_pgm_image(infilename, &image, &rows, &cols) == 0) {
        fprintf(stderr, "Error reading the input image, %s.\n", infilename);
        exit(1);
    }

    /***************************************************************************
    * Perform the edge detection. All of the work takes place here.
    ***************************************************************************/
    if (VERBOSE) printf("Starting Canny edge detection.\n");
    if (dirfilename != NULL) {
        sprintf(composedfname, "output/%s_s_%3.2f_l_%3.2f_h_%3.2f.fim", 
                basename, sigma, tlow, thigh);
        dirfilename = composedfname;
    }

    // Call the GPU-enabled canny function (only Gaussian smoothing is offloaded to GPU)
    cuda_canny(image, rows, cols, sigma, tlow, thigh, &edge, dirfilename);

    /****************************************************************************
     * Write out the edge image to a file.
     ****************************************************************************/
    sprintf(outfilename, "output/cuda_%s_s_%3.2f_l_%3.2f_h_%3.2f.pgm", basename,
        sigma, tlow, thigh);
    if(VERBOSE) printf("Writing the edge iname in the file %s.\n", outfilename);
    if(write_pgm_image(outfilename, edge, rows, cols, "", 255) == 0){
        fprintf(stderr, "Error writing the edge image, %s.\n", outfilename);
        exit(1);
    }
    return 0;
}
