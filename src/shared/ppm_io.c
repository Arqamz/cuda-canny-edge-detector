#include "ppm_io.h"

/******************************************************************************
 * Function: read_ppm_image
 * Purpose: This function reads in an image in PPM format. The image can be
 * read in from either a file or from standard input. The image is only read
 * from standard input when infilename = NULL. Because the PPM format includes
 * the number of columns and the number of rows in the image, these are read
 * from the file. Memory to store the image is allocated in this function.
 * All comments in the header are discarded in the process of reading the
 * image. Upon failure, this function returns 0, upon sucess it returns 1.
 ******************************************************************************/
int read_ppm_image(char *infilename, unsigned char **image_red, unsigned char **image_grn, unsigned char **image_blu, int *rows, int *cols)
{
   FILE *fp;
   char buf[71];
   int p, size;

   /***************************************************************************
    * Open the input image file for reading if a filename was given. If no
    * filename was provided, set fp to read from standard input.
    ***************************************************************************/
   if (infilename == NULL)
      fp = stdin;
   else
   {
      if ((fp = fopen(infilename, "r")) == NULL)
      {
         fprintf(stderr, "Error reading the file %s in read_ppm_image().\n",
                 infilename);
         return (0);
      }
   }

   /***************************************************************************
    * Verify that the image is in PPM format, read in the number of columns
    * and rows in the image and scan past all of the header information.
    ***************************************************************************/
   fgets(buf, 70, fp);
   if (strncmp(buf, "P6", 2) != 0)
   {
      fprintf(stderr, "The file %s is not in PPM format in ", infilename);
      fprintf(stderr, "read_ppm_image().\n");
      if (fp != stdin)
         fclose(fp);
      return (0);
   }
   do
   {
      fgets(buf, 70, fp);
   } while (buf[0] == '#'); /* skip all comment lines */
   sscanf(buf, "%d %d", cols, rows);
   do
   {
      fgets(buf, 70, fp);
   } while (buf[0] == '#'); /* skip all comment lines */

   /***************************************************************************
    * Allocate memory to store the image then read the image from the file.
    ***************************************************************************/
   if (((*image_red) = (unsigned char *)malloc((*rows) * (*cols))) == NULL)
   {
      fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
      if (fp != stdin)
         fclose(fp);
      return (0);
   }
   if (((*image_grn) = (unsigned char *)malloc((*rows) * (*cols))) == NULL)
   {
      fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
      if (fp != stdin)
         fclose(fp);
      return (0);
   }
   if (((*image_blu) = (unsigned char *)malloc((*rows) * (*cols))) == NULL)
   {
      fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
      if (fp != stdin)
         fclose(fp);
      return (0);
   }

   size = (*rows) * (*cols);
   for (p = 0; p < size; p++)
   {
      (*image_red)[p] = (unsigned char)fgetc(fp);
      (*image_grn)[p] = (unsigned char)fgetc(fp);
      (*image_blu)[p] = (unsigned char)fgetc(fp);
   }

   if (fp != stdin)
      fclose(fp);
   return (1);
}

/******************************************************************************
 * Function: write_ppm_image
 * Purpose: This function writes an image in PPM format. The file is either
 * written to the file specified by outfilename or to standard output if
 * outfilename = NULL. A comment can be written to the header if coment != NULL.
 ******************************************************************************/
int write_ppm_image(char *outfilename, unsigned char *image_red, unsigned char *image_grn, unsigned char *image_blu, int rows, int cols, char *comment, int maxval)
{
   FILE *fp;
   long size, p;

   /***************************************************************************
    * Open the output image file for writing if a filename was given. If no
    * filename was provided, set fp to write to standard output.
    ***************************************************************************/
   if (outfilename == NULL)
      fp = stdout;
   else
   {
      if ((fp = fopen(outfilename, "w")) == NULL)
      {
         fprintf(stderr, "Error writing the file %s in write_pgm_image().\n",
                 outfilename);
         return (0);
      }
   }

   /***************************************************************************
    * Write the header information to the PGM file.
    ***************************************************************************/
   fprintf(fp, "P6\n%d %d\n", cols, rows);
   if (comment != NULL)
      if (strlen(comment) <= 70)
         fprintf(fp, "# %s\n", comment);
   fprintf(fp, "%d\n", maxval);

   /***************************************************************************
    * Write the image data to the file.
    ***************************************************************************/
   size = (long)rows * (long)cols;
   for (p = 0; p < size; p++)
   { /* Write the image in pixel interleaved format. */
      fputc(image_red[p], fp);
      fputc(image_grn[p], fp);
      fputc(image_blu[p], fp);
   }

   if (fp != stdout)
      fclose(fp);
   return (1);
}
