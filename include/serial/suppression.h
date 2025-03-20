#ifndef SUPPRESSION_H
#define SUPPRESSION_H

#ifdef __cplusplus
extern "C" {
#endif

#define NOEDGE 255
#define POSSIBLE_EDGE 128
#define EDGE 0

/*******************************************************************************
 * FUNCTION: non_max_supp
 * PURPOSE: This function applies non-maximal suppression to the magnitude of
 * the gradient image.
 * PARAMETERS:
 *   mag - Magnitude values
 *   gradx - X gradient values
 *   grady - Y gradient values
 *   nrows - Number of rows in the image
 *   ncols - Number of columns in the image
 *   result - Output result after non-maximal suppression
 * RETURN: int
 *******************************************************************************/
int non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols, unsigned char *result);

#ifdef __cplusplus
}
#endif

#endif // SUPPRESSION_H