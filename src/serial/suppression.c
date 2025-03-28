#include "suppression.h"

/*******************************************************************************
 * PROCEDURE: non_max_supp
 * PURPOSE: This routine applies non-maximal suppression to the magnitude of
 * the gradient image.
 * NAME: Mike Heath
 * DATE: 2/15/96
 *******************************************************************************/
int non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols, unsigned char *result)
{
    int rowcount, colcount, count;
    short *magrowptr, *magptr;
    short *gxrowptr, *gxptr;
    short *gyrowptr, *gyptr, z1, z2;
    short m00, gx, gy;
    float mag1, mag2, xperp, yperp;
    unsigned char *resultrowptr, *resultptr;

    /****************************************************************************
     * Zero the edges of the result image.
     ****************************************************************************/
    for (count = 0, resultrowptr = result, resultptr = result + ncols * (nrows - 1);
         count < ncols; resultptr++, resultrowptr++, count++)
    {
        *resultrowptr = *resultptr = (unsigned char)0;
    }

    for (count = 0, resultptr = result, resultrowptr = result + ncols - 1;
         count < nrows; count++, resultptr += ncols, resultrowptr += ncols)
    {
        *resultptr = *resultrowptr = (unsigned char)0;
    }

    /****************************************************************************
     * Suppress non-maximum points.
     ****************************************************************************/
    for (rowcount = 1, magrowptr = mag + ncols + 1, gxrowptr = gradx + ncols + 1,
        gyrowptr = grady + ncols + 1, resultrowptr = result + ncols + 1;
         rowcount < nrows - 2;
         rowcount++, magrowptr += ncols, gyrowptr += ncols, gxrowptr += ncols,
        resultrowptr += ncols)
    {
        for (colcount = 1, magptr = magrowptr, gxptr = gxrowptr, gyptr = gyrowptr,
            resultptr = resultrowptr;
             colcount < ncols - 2;
             colcount++, magptr++, gxptr++, gyptr++, resultptr++)
        {
            m00 = *magptr;
            if (m00 == 0)
            {
                *resultptr = (unsigned char)NOEDGE;
            }
            else
            {
                xperp = -(gx = *gxptr) / ((float)m00);
                yperp = (gy = *gyptr) / ((float)m00);
            }

            if (gx >= 0)
            {
                if (gy >= 0)
                {
                    if (gx >= gy)
                    {
                        /* 111 */
                        /* Left point */
                        z1 = *(magptr - 1);
                        z2 = *(magptr - ncols - 1);

                        mag1 = (m00 - z1) * xperp + (z2 - z1) * yperp;

                        /* Right point */
                        z1 = *(magptr + 1);
                        z2 = *(magptr + ncols + 1);

                        mag2 = (m00 - z1) * xperp + (z2 - z1) * yperp;
                    }
                    else
                    {
                        /* 110 */
                        /* Left point */
                        z1 = *(magptr - ncols);
                        z2 = *(magptr - ncols - 1);

                        mag1 = (z1 - z2) * xperp + (z1 - m00) * yperp;

                        /* Right point */
                        z1 = *(magptr + ncols);
                        z2 = *(magptr + ncols + 1);

                        mag2 = (z1 - z2) * xperp + (z1 - m00) * yperp;
                    }
                }
                else
                {
                    if (gx >= -gy)
                    {
                        /* 101 */
                        /* Left point */
                        z1 = *(magptr - 1);
                        z2 = *(magptr + ncols - 1);

                        mag1 = (m00 - z1) * xperp + (z1 - z2) * yperp;

                        /* Right point */
                        z1 = *(magptr + 1);
                        z2 = *(magptr - ncols + 1);

                        mag2 = (m00 - z1) * xperp + (z1 - z2) * yperp;
                    }
                    else
                    {
                        /* 100 */
                        /* Left point */
                        z1 = *(magptr + ncols);
                        z2 = *(magptr + ncols - 1);

                        mag1 = (z1 - z2) * xperp + (m00 - z1) * yperp;

                        /* Right point */
                        z1 = *(magptr - ncols);
                        z2 = *(magptr - ncols + 1);

                        mag2 = (z1 - z2) * xperp + (m00 - z1) * yperp;
                    }
                }
            }
            else
            {
                if ((gy = *gyptr) >= 0)
                {
                    if (-gx >= gy)
                    {
                        /* 011 */
                        /* Left point */
                        z1 = *(magptr + 1);
                        z2 = *(magptr - ncols + 1);

                        mag1 = (z1 - m00) * xperp + (z2 - z1) * yperp;

                        /* Right point */
                        z1 = *(magptr - 1);
                        z2 = *(magptr + ncols - 1);

                        mag2 = (z1 - m00) * xperp + (z2 - z1) * yperp;
                    }
                    else
                    {
                        /* 010 */
                        /* Left point */
                        z1 = *(magptr - ncols);
                        z2 = *(magptr - ncols + 1);

                        mag1 = (z2 - z1) * xperp + (z1 - m00) * yperp;

                        /* Right point */
                        z1 = *(magptr + ncols);
                        z2 = *(magptr + ncols - 1);

                        mag2 = (z2 - z1) * xperp + (z1 - m00) * yperp;
                    }
                }
                else
                {
                    if (-gx > -gy)
                    {
                        /* 001 */
                        /* Left point */
                        z1 = *(magptr + 1);
                        z2 = *(magptr + ncols + 1);

                        mag1 = (z1 - m00) * xperp + (z1 - z2) * yperp;

                        /* Right point */
                        z1 = *(magptr - 1);
                        z2 = *(magptr - ncols - 1);

                        mag2 = (z1 - m00) * xperp + (z1 - z2) * yperp;
                    }
                    else
                    {
                        /* 000 */
                        /* Left point */
                        z1 = *(magptr + ncols);
                        z2 = *(magptr + ncols + 1);

                        mag1 = (z2 - z1) * xperp + (m00 - z1) * yperp;

                        /* Right point */
                        z1 = *(magptr - ncols);
                        z2 = *(magptr - ncols - 1);

                        mag2 = (z2 - z1) * xperp + (m00 - z1) * yperp;
                    }
                }
            }

            /* Now determine if the current point is a maximum point */

            if ((mag1 > 0.0) || (mag2 > 0.0))
            {
                *resultptr = (unsigned char)NOEDGE;
            }
            else
            {
                if (mag2 == 0.0)
                    *resultptr = (unsigned char)NOEDGE;
                else
                    *resultptr = (unsigned char)POSSIBLE_EDGE;
            }
        }
    }
}
