#ifndef PPM_IO_H
#define PPM_IO_H

/**
 * @file ppm_io.h
 * @brief Functions for reading and writing PPM image files
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Reads an image in PPM format
 *
 * This function reads in an image in PPM format. The image can be
 * read in from either a file or from standard input. The image is only read
 * from standard input when infilename = NULL. Because the PPM format includes
 * the number of columns and the number of rows in the image, these are read
 * from the file. Memory to store the image is allocated in this function.
 * All comments in the header are discarded in the process of reading the
 * image. Upon failure, this function returns 0, upon success it returns 1.
 *
 * @param infilename Name of the file to read, or NULL for stdin
 * @param image_red Pointer to memory location where red channel data will be stored
 * @param image_grn Pointer to memory location where green channel data will be stored
 * @param image_blu Pointer to memory location where blue channel data will be stored
 * @param rows Pointer to variable where the number of rows will be stored
 * @param cols Pointer to variable where the number of columns will be stored
 * @return 1 on success, 0 on failure
 */
int read_ppm_image(char *infilename, unsigned char **image_red, unsigned char **image_grn, unsigned char **image_blu, int *rows, int *cols);

/**
 * @brief Writes an image in PPM format
 *
 * This function writes an image in PPM format. The file is either
 * written to the file specified by outfilename or to standard output if
 * outfilename = NULL. A comment can be written to the header if coment != NULL.
 *
 * @param outfilename Name of the file to write, or NULL for stdout
 * @param image_red Pointer to red channel image data
 * @param image_grn Pointer to green channel image data
 * @param image_blu Pointer to blue channel image data
 * @param rows Number of rows in the image
 * @param cols Number of columns in the image
 * @param comment Comment to include in the PPM header, or NULL for no comment
 * @param maxval Maximum pixel value (typically 255)
 * @return 1 on success, 0 on failure
 */
int write_ppm_image(char *outfilename, unsigned char *image_red, unsigned char *image_grn, unsigned char *image_blu, int rows, int cols, char *comment, int maxval);

#endif /* PPM_IO_H */