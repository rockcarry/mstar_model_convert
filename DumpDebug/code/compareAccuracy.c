#include <stdio.h>
#include <math.h>

typedef int BOOL;
#define TRUE  (1)
#define FALSE (0)
#define max(a,b)    (((a) > (b)) ? (a) : (b))
#define min(a,b)    (((a) < (b)) ? (a) : (b))
BOOL bFloat = TRUE;
static char op_type[128] = {0};
static char op_data_type[128] = {0};

#define MAX_SHAPE_LEN (4)



BOOL getBufferSize(char *buf, int *buffer_size)
{
    char *p = NULL;

    p = strchr(buf, '/');
    if (p == NULL)
        return FALSE;

    sscanf(p, "//buffer data size: %d", buffer_size);

    if (*buffer_size == 0)
        return FALSE;

    return TRUE;
}

BOOL getOP_TYPE(char *buf, char *op_type)
{
    char *p = NULL;
    int num = 0;
    p = strchr(buf, 'op_out[');
    if (p == NULL)
        return FALSE;

    sscanf(p, "[%d] %s", &num, op_type);

    if (*op_type == 0)
        return FALSE;

    return TRUE;
}

BOOL getOP_Data_Tpye(char *buf, char *op_data_type)
{
    char *p = NULL;
    int num = 0;
    p = strstr(buf, "type:");
    if (p == NULL)
        return FALSE;
    sscanf(p,"type: %s",op_data_type);

    if (*op_data_type == 0)
        return FALSE;

    return TRUE;
}

BOOL getOP_Tensor_Dims(char *buf, int* dims)
{
    char *p = NULL;
    int num = 0;
    p = strstr(buf, "dims:");
    if (p == NULL)
        return FALSE;

    sscanf(p, "dims:%d", dims);

    if (*dims == 0)
        return FALSE;

    return TRUE;
}

BOOL getOP_4dims_Tensor_Shape(char *buf, int *tensorShape)
{
    char *p = NULL;
    int num = 0;
    p = strstr(buf, "shape:");
    if (p == NULL)
        return FALSE;

    sscanf(p, "shape:[%d, %d, %d, %d]", &tensorShape[0], &tensorShape[1], &tensorShape[2], &tensorShape[3]);

    if (*tensorShape == 0)
        return FALSE;

    return TRUE;
}

typedef enum {
    E_TRANSPOSE_MODE_NHWC2NCHW   = 0,
    E_TRANSPOSE_MODE_NCHW2NHWC   = 1,
    E_TRANSPOSE_MODE_MAX,
} EN_TransposeMode_e;

int SGS_OPS_CalculateTensorDataOffset(int as32Shape[], int as32DimIndex[], int s32DimSize)
{
    int s32DataIndex = as32DimIndex[0];
    for(int s32Index = 0; s32Index < s32DimSize - 1; s32Index ++)
    {
        s32DataIndex = (s32DataIndex) * as32Shape[s32Index+1] + as32DimIndex[s32Index+1];
    }
    return s32DataIndex;
}

BOOL Transpose4dims(EN_TransposeMode_e eTransposeMode,
    char* pu8Input, char* pu8Output, int  asTensorShape[])
{
    BOOL bRet = TRUE;
    char* pu8InputData = pu8Input;
    char* pu8OutputData = pu8Output;
    int as32Out[MAX_SHAPE_LEN] = {0};  // loop index (on output).
    int as32In[MAX_SHAPE_LEN] = {0};
    int as32ExtendedPerm[MAX_SHAPE_LEN] = {0};
    int as32ExtendedInShape[MAX_SHAPE_LEN] = {0};
    int as32ExtendedOutShape[MAX_SHAPE_LEN] = {0};
    int s32DataIdxIn = 0;
    int s32DataIdxOut = 0;
    int s32InnermostShape = 0;
    int s32AlignedInnermostShape = 0;
    int s32AlignmentSize = asTensorShape[0]*asTensorShape[1]*asTensorShape[2]*asTensorShape[3];
    int s32ElementSize = sizeof(float);


    memcpy(as32ExtendedInShape, asTensorShape, sizeof(int) * 4);
    if(eTransposeMode == E_TRANSPOSE_MODE_NCHW2NHWC)
    {
        as32ExtendedPerm[0] = 0;
        as32ExtendedPerm[1] = 2;
        as32ExtendedPerm[2] = 3;
        as32ExtendedPerm[3] = 1;
    }
    else
    {
        as32ExtendedPerm[0] = 0;
        as32ExtendedPerm[1] = 3;
        as32ExtendedPerm[2] = 1;
        as32ExtendedPerm[3] = 2;
    }
    as32ExtendedOutShape[0] = asTensorShape[as32ExtendedPerm[0]];
    as32ExtendedOutShape[1] = asTensorShape[as32ExtendedPerm[1]];
    as32ExtendedOutShape[2] = asTensorShape[as32ExtendedPerm[2]];
    as32ExtendedOutShape[3] = asTensorShape[as32ExtendedPerm[3]];

    s32InnermostShape = as32ExtendedOutShape[MAX_SHAPE_LEN - 1];
    s32AlignedInnermostShape = s32InnermostShape;

    for(as32Out[0] = 0; as32Out[0] < as32ExtendedOutShape[0]; as32Out[0]++)
    {
        as32In[as32ExtendedPerm[0]] = as32Out[0];
        for(as32Out[1] = 0; as32Out[1] < as32ExtendedOutShape[1]; as32Out[1]++)
        {
            as32In[as32ExtendedPerm[1]] = as32Out[1];
            for(as32Out[2] = 0; as32Out[2] < as32ExtendedOutShape[2]; as32Out[2]++)
            {
                as32In[as32ExtendedPerm[2]] = as32Out[2];
                for(as32Out[3] = 0; as32Out[3] < s32InnermostShape; as32Out[3]++)
                {
                    as32In[as32ExtendedPerm[3]] = as32Out[3];
                    s32DataIdxIn = SGS_OPS_CalculateTensorDataOffset(as32ExtendedInShape,
                                as32In, MAX_SHAPE_LEN) * s32ElementSize;

                    memcpy(&pu8OutputData[s32DataIdxOut], &pu8InputData[s32DataIdxIn],
                        s32ElementSize);

                    if((as32Out[3] == s32InnermostShape-1)
                        && (s32InnermostShape != s32AlignedInnermostShape))
                    {
                        s32DataIdxOut += s32ElementSize
                                      * (s32AlignedInnermostShape-s32InnermostShape+1);
                    }
                    else
                    {
                        s32DataIdxOut += s32ElementSize;
                    }
                }
            }
        }
    }

    return bRet;
}

BOOL doCompare(FILE *file_1,FILE *file_2)
{
    static char buffer0[512];
    char *pBuffer1 = NULL;
    char *pBuffer2 = NULL;
    char *pBuffer2_tmp = NULL;
    int file2_buffer_size = 0;
    int file1_buffer_size = 0;
    int i = 0;
    int file1_dims = 0;
    int file1_tensor_shape[4] = {0};
    int file2_dims = 0;
    int file2_tensor_shape[4] = {0};
    char file1_op_type[128] = {0};
    char file1_op_data_type[128] = {0};
    int file2_new_tensor_shape[4] = {0};
    BOOL is_NCHW_NHWC_equivalent = FALSE;
    //float out[24] = {0};

    for(i = 0; i<24; i++)
    {
        //printf("in %f\n",in[i]);
    }
    for(i = 0; i<24; i++)
    {
        //printf("out %f\n",out[i]);
    }

    memset(buffer0, 0, sizeof(buffer0));
    fgets(buffer0, sizeof(buffer0), file_2);
    sscanf(buffer0, "isFloat: %d", &bFloat);
    /*
        benchmark file(file_2) struct
        isFloat: 1
        //out 0 s: 0.000239 z: 0 type: SGS_S16 name: 215 bConstant:0 shape:[3, 1, 3, 197, 64] dims:5
        op_out[11] TRANSPOSE = {
        //buffer data size: 453888
    */

    do
    {
        memset(buffer0, 0, sizeof(buffer0));
        fgets(buffer0, sizeof(buffer0), file_2);
        getOP_TYPE(buffer0, &op_type);
        getOP_Data_Tpye (buffer0, &op_data_type);
        getOP_Tensor_Dims (buffer0, &file2_dims);
        if(file2_dims == 4)
        {
            getOP_4dims_Tensor_Shape (buffer0, file2_tensor_shape);
        }
        if(getBufferSize(buffer0, &file2_buffer_size)) break;
    }while(1);

    do
    {
        memset(buffer0, 0, sizeof(buffer0));
        fgets(buffer0, sizeof(buffer0), file_1);
        getOP_TYPE(buffer0, &file1_op_type);
        getOP_Data_Tpye (buffer0, &file1_op_data_type);
        getOP_Tensor_Dims (buffer0, &file1_dims);
        if(file1_dims == 4)
        {
            getOP_4dims_Tensor_Shape (buffer0, file1_tensor_shape);
        }
        if(getBufferSize(buffer0, &file1_buffer_size)) break;
    }while(1);

    if(file1_buffer_size != file2_buffer_size)
    {
        printf("Error! file1 sample buffer size %d != file2 benchmark buffer size %d, when op type is %s\n",file1_buffer_size, file2_buffer_size, op_type);
        return 0;
    }

    //the dims number of sim model and caffe/onnx model might be different but with same buffer size.
    //So don't assert, compare as usual when dims are not same.
    //if(file1_dims != file2_dims)
    //{
    //    printf("Error! file1 sample dims %d != file2 benchmark dims %d, when op type is %s\n",file1_dims, file2_dims, op_type);
    //    return 0;
    //}

    pBuffer1 = (char*)calloc(1,file2_buffer_size);
    fread(pBuffer1, file2_buffer_size,1, file_1);

    pBuffer2 = (char*)calloc(1,file2_buffer_size);
    fread(pBuffer2, file2_buffer_size,1, file_2);

    //note file2 NHWC-NCHW transposed shape
    file2_new_tensor_shape[0] = file2_tensor_shape[0];
    file2_new_tensor_shape[1] = file2_tensor_shape[3];
    file2_new_tensor_shape[2] = file2_tensor_shape[1];
    file2_new_tensor_shape[3] = file2_tensor_shape[2];

    if(file2_dims == 4 && memcmp(file1_tensor_shape, file2_tensor_shape, sizeof(int)*4))
    {
        //when dims=4 and shapes not same, try NHWC-NCHW transpose and check shapes again.
        is_NCHW_NHWC_equivalent = FALSE;

        if(memcmp(file1_tensor_shape, file2_new_tensor_shape, sizeof(int)*4))
        {
            printf("Error! file1 sample shape[%d,%d,%d,%d] != file2 benchmark NHWC shape[%d,%d,%d,%d] or NCHW shape [%d,%d,%d,%d], when op type is %s\n",
                file1_tensor_shape[0],
                file1_tensor_shape[1],
                file1_tensor_shape[2],
                file1_tensor_shape[3],
                file2_tensor_shape[0],
                file2_tensor_shape[1],
                file2_tensor_shape[2],
                file2_tensor_shape[3],
                file2_new_tensor_shape[0],
                file2_new_tensor_shape[1],
                file2_new_tensor_shape[2],
                file2_new_tensor_shape[3],
                op_type);
            free(pBuffer1);
            free(pBuffer2);
            return 0;
        }
        else
        {
            pBuffer2_tmp = (char*)calloc(1,file2_buffer_size);
            Transpose4dims(E_TRANSPOSE_MODE_NHWC2NCHW, pBuffer2, pBuffer2_tmp, file2_tensor_shape);
            memcpy(pBuffer2, pBuffer2_tmp, file2_buffer_size);
            free(pBuffer2_tmp);
        }
    }
    else
    {
        //when dims=4 and shapes are same but with NCHW/NHWC equivalent shape (like 1,18,18,18)
        //try NHWC-NCHW transpose in compareNum if COS result is bad.
        if(file2_dims == 4 && !memcmp(file2_tensor_shape, file2_new_tensor_shape, sizeof(int)*4))
        {
            is_NCHW_NHWC_equivalent = TRUE;
        }
    }

    compareNum(pBuffer1, pBuffer2, file2_buffer_size, file2_dims, file2_tensor_shape, is_NCHW_NHWC_equivalent);
    free(pBuffer1);
    free(pBuffer2);
    return 0;


}

BOOL compareNum(void *op_out_force_data,void *op_out_force_data2,int buffer_size, int file2_dims, int file2_shape[], BOOL is_NCHW_NHWC_equivalent)
{
    int size = 0;
    float *pData = (float *)op_out_force_data;
    float *pData2 = (float *)op_out_force_data2;
    float *pData2_temp = NULL;

    if(bFloat)
    {
      size = buffer_size/4;
    }
    else
    {
      size = buffer_size;
    }
    double fmin = 0;
    double fmax = 0;

    int amax = 0;
    int index = 0;
    int i = 0;


    amax = size;

    double diff = 0.0, rdiff = 0.0;
    double diffSum = 0.0;
    double rdiffSum = 0.0;
    double rSum = 0.0;

    double muldot = 0.0;
    double muldotSum = 0.0;
    double sq0 = 0.0;
    double sq1 = 0.0;
    double sq0Sum = 0.0;
    double sq1Sum = 0.0;
    double mulsq = 0.0;
    double cos = 0.0;
    double cos2 = 0.0;


    for (i = 0; i < size; i++)
    {
        diff = pData[i] - pData2[i];
        muldot = pData[i] * pData2[i];
        muldotSum += muldot;
        sq0 = pow(pData[i], 2);
        sq1 = pow(pData2[i], 2);
        sq0Sum += sq0;
        sq1Sum += sq1;
        diff =fabs(diff);
        rdiffSum += diff;
        diff = diff*diff;
        diffSum += diff;
        //printf("array[%d] = %f --> %f diff: %f\n", i, pData[i], pData2[i], diff);
        rSum += fabs(pData2[i]);

    }
    mulsq = sqrt(sq0Sum) * sqrt(sq1Sum);
    cos = muldotSum / mulsq;
    diffSum  = diffSum/amax;
    //rdiffSum = rdiffSum/amax;
    rdiffSum   = rdiffSum/rSum;

    //when dims=4 and shapes are same but with NCHW/NHWC equivalent shape (like 1,18,18,18)
    //try NHWC-NCHW transpose if COS result is bad.
    if(cos < 0.98 && file2_dims == 4 && is_NCHW_NHWC_equivalent)
    {
        diff = 0.0, rdiff = 0.0;
        diffSum = 0.0;
        rdiffSum = 0.0;
        rSum = 0.0;

        muldot = 0.0;
        muldotSum = 0.0;
        sq0 = 0.0;
        sq1 = 0.0;
        sq0Sum = 0.0;
        sq1Sum = 0.0;
        mulsq = 0.0;
        cos2 = 0.0;

        pData2_temp = (float*)calloc(1,buffer_size);
        Transpose4dims(E_TRANSPOSE_MODE_NHWC2NCHW, (char*)pData2, (char*)pData2_temp, file2_shape);
        memcpy(pData2, pData2_temp, buffer_size);

        for (i = 0; i < size; i++)
        {
            diff = pData[i] - pData2[i];
            muldot = pData[i] * pData2[i];
            muldotSum += muldot;
            sq0 = pow(pData[i], 2);
            sq1 = pow(pData2[i], 2);
            sq0Sum += sq0;
            sq1Sum += sq1;
            diff =fabs(diff);
            rdiffSum += diff;
            diff = diff*diff;
            diffSum += diff;
            //printf("array[%d] = %f --> %f diff: %f\n", i, pData[i], pData2[i], diff);
            rSum += fabs(pData2[i]);

        }
        mulsq = sqrt(sq0Sum) * sqrt(sq1Sum);
        cos2 = muldotSum / mulsq;
        diffSum  = diffSum/amax;
        //rdiffSum = rdiffSum/amax;
        rdiffSum   = rdiffSum/rSum;

        //choose better COS result to display
        if(cos >= cos2)
        {
            printf("OP: %s  Type: %s  MSE: %f  COS: %f  RMSE:\t%f\n", op_type, op_data_type, diffSum, cos, rdiffSum);

        }
        else
        {
            printf("OP: %s  Type: %s <<origin COS %f < 0.98, file2 NHWC-NCHW transposed>> MSE: %f  COS: %f  RMSE:\t%f\n", op_type, op_data_type, cos, diffSum, cos2, rdiffSum);
        }
    }
    else
    {
        printf("OP: %s  Type: %s  MSE: %f  COS: %f  RMSE:\t%f\n", op_type, op_data_type, diffSum, cos, rdiffSum);
    }

	return 0;
}

int main(int argc, const char *argv[])
{
    if (argc < 3)
    {
        printf("%s <filename>\n", argv[0]);
        return 0;
    }
    FILE * file_1 = NULL; //sample.bin
    FILE * file_2 = NULL; //banchmark.bin

    file_1 = fopen(argv[1], "rb");
    file_2 = fopen(argv[2], "rb");
    if (file_1 != NULL && file_2 != NULL)
    {
        // xxx
        doCompare(file_1,file_2);
        fclose(file_1);
        fclose(file_2);
    }
    else
    {
        perror("fopen");
    }
    return 0;
}

