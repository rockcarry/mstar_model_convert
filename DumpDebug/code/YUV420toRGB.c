/* Copyright (c) 2018-2019 Sigmastar Technology Corp.
 All rights reserved.

  Unless otherwise stipulated in writing, any and all information contained
 herein regardless in any format shall remain the sole proprietary of
 Sigmastar Technology Corp. and be kept in strict confidence
 (��Sigmastar Confidential Information��) by the recipient.
 Any unauthorized act including without limitation unauthorized disclosure,
 copying, use, reproduction, sale, distribution, modification, disassembling,
 reverse engineering and compiling of the contents of Sigmastar Confidential
 Information is unlawful and strictly prohibited. Sigmastar hereby reserves the
 rights to any and all damages, losses, costs and expenses resulting therefrom.
*/


#include <stdio.h>
#include <stdlib.h>

/*
******************************************************
R2Y/Y2R formule:

 R = | 1  0          1613/1024 |   | Y    |
 G = | 1  -192/1024  -479/1024 | * | U-128|
 B = | 1  1900/1024  0         |   | V-128|

 R =  Y + 0                   + 1613/1024 * (V-128)
 G =  Y - 192/1024  * (U-128) - 479/1024  * (V-128)
 B =  Y + 1900/1024 * (U-128) + 0

 Y = (R * 218  + G * 732  + B * 74) / 1024
 U = (R * -117 + G * -395 + B * 512)/ 1024  + 128
 V = (R * 512  + G * -465 + B * -47)/ 1024  + 128
*******************************************************
YUV NV12 420 semiplanar data format:

Y00     Y01   .........   Y0(W-2)     Y0(W-1)
Y10     Y11   .........   Y1(W-2)     Y0(W-1)
                    .
                    .
Y(H-1)0 Y(H-1)1 ......... Y(H-1)(W-2) Y(H-1)(W-1)
U0      V0      ......... U(W/2-1)    V(W/2-1)
                    .
                    .

*******************************************************
*/

static float yuv2rgb_covert_matrix[9] = {1, 0, 1613.0/1024, 1, -192.0/1024, -479.0/1024, 1, 1900.0/1024, 0};

void SGS_yuvNv12ToRgb(float* pYuvBuf, float* pRgbBuf, int height, int width)
{
    if(width < 1 || height < 1 || pYuvBuf == NULL || pRgbBuf == NULL)
    {
        printf("input error\n");
        return;
    }

    const long len = height * width;

    // Y and UV data addr
    float *yData = pYuvBuf;
    float *uvData = yData + len;

    // R、G、B planar data addr
    float *tempRGB = (float *)malloc(height*width*3*sizeof(float));
    float *rData = tempRGB;
    float *gData = rData + len;
    float *bData = gData + len;

    float R[4], G[4], B[4];
    float Y[4], U, V;
    int y0_Idx, y1_Idx, uIdx, vIdx;

    for (int i = 0; i < height; i=i+2)
    {
        for (int j = 0; j < width; j=j+2)
        {
            y0_Idx = i * width + j;
            y1_Idx = (i + 1) * width + j;

            Y[0] = (float)yData[y0_Idx];       //Y00
            Y[1] = (float)yData[y0_Idx + 1];   //Y01
            Y[2] = (float)yData[y1_Idx];       //Y10
            Y[3] = (float)yData[y1_Idx + 1];   //Y11

            uIdx = (i / 2) * width + j;
            vIdx = uIdx + 1;

            U = uvData[uIdx];
            V = uvData[vIdx];

            R[0] = Y[0] * yuv2rgb_covert_matrix[0] + (U - 128) * yuv2rgb_covert_matrix[1] + (V - 128) * yuv2rgb_covert_matrix[2];
            G[0] = Y[0] * yuv2rgb_covert_matrix[3] + (U - 128) * yuv2rgb_covert_matrix[4] + (V - 128) * yuv2rgb_covert_matrix[5];
            B[0] = Y[0] * yuv2rgb_covert_matrix[6] + (U - 128) * yuv2rgb_covert_matrix[7] + (V - 128) * yuv2rgb_covert_matrix[8];

            R[1] = Y[1] * yuv2rgb_covert_matrix[0] + (U - 128) * yuv2rgb_covert_matrix[1] + (V - 128) * yuv2rgb_covert_matrix[2];
            G[1] = Y[1] * yuv2rgb_covert_matrix[3] + (U - 128) * yuv2rgb_covert_matrix[4] + (V - 128) * yuv2rgb_covert_matrix[5];
            B[1] = Y[1] * yuv2rgb_covert_matrix[6] + (U - 128) * yuv2rgb_covert_matrix[7] + (V - 128) * yuv2rgb_covert_matrix[8];

            R[2] = Y[2] * yuv2rgb_covert_matrix[0] + (U - 128) * yuv2rgb_covert_matrix[1] + (V - 128) * yuv2rgb_covert_matrix[2];
            G[2] = Y[2] * yuv2rgb_covert_matrix[3] + (U - 128) * yuv2rgb_covert_matrix[4] + (V - 128) * yuv2rgb_covert_matrix[5];
            B[2] = Y[2] * yuv2rgb_covert_matrix[6] + (U - 128) * yuv2rgb_covert_matrix[7] + (V - 128) * yuv2rgb_covert_matrix[8];

            R[3] = Y[3] * yuv2rgb_covert_matrix[0] + (U - 128) * yuv2rgb_covert_matrix[1] + (V - 128) * yuv2rgb_covert_matrix[2];
            G[3] = Y[3] * yuv2rgb_covert_matrix[3] + (U - 128) * yuv2rgb_covert_matrix[4] + (V - 128) * yuv2rgb_covert_matrix[5];
            B[3] = Y[3] * yuv2rgb_covert_matrix[6] + (U - 128) * yuv2rgb_covert_matrix[7] + (V - 128) * yuv2rgb_covert_matrix[8];

            // Keep RGB data in 0-255
            for (int k = 0; k < 4; ++k)
            {
                if(R[k] >= 0 && R[k] <= 255)
                {
                    R[k] = R[k];
                }
                else
                {
                    R[k] = (R[k] < 0) ? 0 : 255;
                }

                if(G[k] >= 0 && G[k] <= 255)
                {
                    G[k] = G[k];
                }
                else
                {
                    G[k] = (G[k] < 0) ? 0 : 255;
                }

                if(B[k] >= 0 && B[k] <= 255)
                {
                    B[k] = B[k];
                }
                else
                {
                    B[k] = (B[k] < 0) ? 0 : 255;
                }
            }

            // write R、G、B planar data
            *(rData + y0_Idx) = R[0];
            *(gData + y0_Idx) = G[0];
            *(bData + y0_Idx) = B[0];

            *(rData + y0_Idx + 1) = R[1];
            *(gData + y0_Idx + 1) = G[1];
            *(bData + y0_Idx + 1) = B[1];

            *(rData + y1_Idx) = R[2];
            *(gData + y1_Idx) = G[2];
            *(bData + y1_Idx) = B[2];

            *(rData + y1_Idx + 1) = R[3];
            *(gData + y1_Idx + 1) = G[3];
            *(bData + y1_Idx + 1) = B[3];
        }
    }

    // convert RGB planar to RGB packed
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            *(pRgbBuf + (i * width + j) * 3) = *(rData + (i * width + j));
            *(pRgbBuf + (i * width + j) * 3 +1) = *(gData + (i * width + j));
            *(pRgbBuf + (i * width + j) * 3 +2) = *(bData + (i * width + j));
        }
    }
    return;
}

int main(int argc, char **argv)
{
    char outputPath[] = "output.RGB";
    int height = 0;
    int width = 0;
    int i = 0;
    FILE *fp;
    FILE *fp2;
    unsigned char ch;

    if(argc < 4)
    {
        printf("Error input! should be: ./YUV420toRGB yuvFileName height width\n");
        exit(1);
    }
    height = (int)strtol(argv[2], NULL, 10);
    width = (int)strtol(argv[3], NULL, 10);

    float *pYuvBufFloat = (float *)malloc(height*width*3*sizeof(float)/2);
    float *pRgbBufFloat = (float *)malloc(height*width*3*sizeof(float));
    unsigned char *pRgbBufUint8 = (unsigned char *)malloc(height*width*3*sizeof(char));

    if (!(fp=fopen(argv[1],"r")))
    {
        printf("Error in open YUV file!\n");
        exit(1);
    }

    for (i=0;i<height*width*3/2;i++)
    {
        ch = fgetc(fp);

        if(ch == EOF)
        {
            break;
        }
        else
        {
            *(pYuvBufFloat + i)=(float)ch;
            //printf("%f ",*(pYuvBufFloat + i));
        }
    }

    //printf("\n i = %d\n",i);
    fclose(fp);

    remove(outputPath);
    if (!(fp2=fopen(outputPath,"a")))
    {
        printf("Error in open file output.RGB!\n");
        exit(1);
    }

    SGS_yuvNv12ToRgb(pYuvBufFloat, pRgbBufFloat, height, width);

    for (i=0;i<height*width*3;i++)
    {
        *(pRgbBufUint8+i) = (unsigned char)*(pRgbBufFloat+i);
        fputc(*(pRgbBufUint8+i), fp2);
    }

    printf("YUV420 NV12 semi planar to RGB done!\n Output RGB file is output.RGB. height=%d width=%d\n",height,width);

    fclose(fp2);
    free(pYuvBufFloat);
    free(pRgbBufFloat);
    free(pRgbBufUint8);

    return 0;
}


