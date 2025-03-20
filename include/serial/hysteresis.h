/*******************************************************************************
 * FILE: hysteresis.h
 * This is the header file for hysteresis.c
 * Original code by Mike Heath
 *******************************************************************************/
#ifndef HYSTERESIS_H

#define HYSTERESIS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Edge status constants */
#define NOEDGE 255
#define POSSIBLE_EDGE 128
#define EDGE 0

#include <stdio.h>
#include <stdlib.h>

/*******************************************************************************
 * FUNCTION: follow_edges
 * PURPOSE: This function traces edges along all paths whose magnitude values remain
 * above some specifiable lower threshold.
 * PARAMETERS:
 *   edgemapptr - Pointer to the edge map
 *   edgemagptr - Pointer to the edge magnitude
 *   lowval - Low threshold value
 *   cols - Number of columns in the image
 * RETURN: int
 *******************************************************************************/
int follow_edges(unsigned char *edgemapptr, short *edgemagptr, short lowval, int cols);

/*******************************************************************************
 * FUNCTION: apply_hysteresis
 * PURPOSE: This function finds edges that are above some high threshold or
 * are connected to a high pixel by a path of pixels greater than a low threshold.
 * PARAMETERS:
 *   mag - Magnitude values
 *   nms - Non-maximal suppression result
 *   rows - Number of rows in the image
 *   cols - Number of columns in the image
 *   tlow - Low threshold fraction
 *   thigh - High threshold fraction
 *   edge - Output edge map
 * RETURN: void
 *******************************************************************************/
void apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols, float tlow, float thigh, unsigned char *edge);

#ifdef __cplusplus
}
#endif

#endif /* HYSTERESIS_H */