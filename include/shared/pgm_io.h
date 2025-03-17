/*******************************************************************************
 * FILE: pgm_io.h
 * This code was adapted from Mike Heath. heath@csee.usf.edu (1995).
 *******************************************************************************/

#ifndef PGM_IO_H
#define PGM_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/******************************************************************************
 * Function: read_pgm_image
 * Purpose: This function reads in an image in PGM format. The image can be
 * read in from either a file or from standard input. The image is only read
 * from standard input when infilename = NULL. Because the PGM format includes
 * the number of columns and the number of rows in the image, these are read
 * from the file. Memory to store the image is allocated in this function.
 * All comments in the header are discarded in the process of reading the
 * image. Upon failure, this function returns 0, upon success it returns 1.
 ******************************************************************************/
int read_pgm_image(char *infilename, unsigned char **image, int *rows, int *cols);

/******************************************************************************
 * Function: write_pgm_image
 * Purpose: This function writes an image in PGM format. The file is either
 * written to the file specified by outfilename or to standard output if
 * outfilename = NULL. A comment can be written to the header if comment != NULL.
 ******************************************************************************/
int write_pgm_image(char *outfilename, unsigned char *image, int rows, int cols, char *comment, int maxval);

#endif /* PGM_IO_H */