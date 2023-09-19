#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

typedef unsigned char      BOOL;
typedef unsigned char      U8;
typedef signed char        S8;
typedef unsigned short     U16;
typedef signed short       S16;
typedef unsigned int       U32;
typedef signed int         S32;
typedef unsigned long long U64;
typedef signed long long   S64;

#define TRUE  (1)
#define FALSE (0)
static char *file_in_name = "input";
static char *file_out_name = NULL;//"output";
BOOL bFloat = TRUE;

typedef enum {
    E_DATA_TYPE_FLOAT = 0,
    E_DATA_TYPE_INT8,
    E_DATA_TYPE_UINT8,
    E_DATA_TYPE_INT16,
    E_DATA_TYPE_UINT16,
    E_DATA_TYPE_INT32,
    E_DATA_TYPE_BOOL,
    E_DATA_TYPE_MAX,
} ST_DataType_e;

ST_DataType_e getDataTypeByName(const char* pstrDataName)
{
    ST_DataType_e eDataType= E_DATA_TYPE_MAX;
    if (strcmp(pstrDataName, "SGS_S16") == 0)
    {
        eDataType = E_DATA_TYPE_INT16;
    }
    else if (strcmp(pstrDataName, "SGS_S8") == 0)
    {
        eDataType = E_DATA_TYPE_INT8;
    }
//    else if (strcmp(pstrDataName, "SGS_U8") == 0)
//    {
//        eDataType = E_DATA_TYPE_UINT8;
//    }
//    else if (strcmp(pstrDataName, "float") == 0)
//    {
//        eDataType = E_DATA_TYPE_FLOAT;
//    }
//    else if (strcmp(pstrDataName, "SGS_U16") == 0)
//    {
//        eDataType = E_DATA_TYPE_UINT16;
//    }
//    else if (strcmp(pstrDataName, "SGS_S32") == 0)
//    {
//        eDataType = E_DATA_TYPE_INT32;
//    }
//    else if (strcmp(pstrDataName, "SGS_BOOL") == 0)
//    {
//        eDataType = E_DATA_TYPE_BOOL;
//    }
    else
    {
        eDataType = E_DATA_TYPE_UINT8;
    }

    return eDataType;
}


char *strrpc(char *str,char *oldstr,char *newstr){
    char bstr[strlen(str)];
    memset(bstr,0,sizeof(bstr));

    for(int i = 0;i < strlen(str);i++){
        if(!strncmp(str+i,oldstr,strlen(oldstr))){
            strcat(bstr,newstr);
            i += strlen(oldstr) - 1;
        }else{
           strncat(bstr,str + i,1);
        }
    }

    strcpy(str,bstr);
    return str;
}

BOOL isHeadLine0(char *buf, char *pfile_name, int *output_idx, ST_DataType_e *peDataType)
{
    char *p = NULL;
    char in_out_str[8];
    int file_num = 0;
    float scale = 0;
    S64 s64ZeroPoint = 0;
    char dataName[10];
    p = strchr(buf, '/');
    if (p == NULL)
        return FALSE;

    memset(in_out_str, 0, sizeof(in_out_str));
    memset(dataName, 0, sizeof(dataName));
    sscanf(p, "//%s %d s: %f z: %lld type: %s name", in_out_str, &file_num, &scale, &s64ZeroPoint, dataName);

    if (strcmp("in", in_out_str) == 0)
        sprintf(pfile_name, "%s%d",  file_in_name, file_num);
    else if (strcmp("out", in_out_str) == 0) {
        //sprintf(pfile_name, "%s%d",  file_out_name, file_num);
        sprintf(pfile_name, "%s",  file_out_name);
        *output_idx = file_num;
    }
    else if (strcmp("buffer", in_out_str) == 0) {
        return FALSE;
    }
    else
    {
        perror("headline0 but no input or output");
        return FALSE;
    }
    *peDataType = getDataTypeByName(dataName);

    return TRUE;
}

BOOL isHeadLine1(char *buf, char *pdir_name, int output_idx, char *buf0)
{
    char *p = NULL;
    int opnode_Num = -1;
    char opnode_name[64];
    char tensorName[256];

    //find tensor name
    p = strstr(buf0, "name:");
    if (p == NULL)
    {
        return FALSE;
    }
    memset(tensorName, 0, sizeof(tensorName));
    sscanf(p,"name: %s",tensorName);
    strrpc(tensorName, "/", "_");
    strrpc(tensorName, ":", "");
    memset(opnode_name, 0, sizeof(opnode_name));
    p = strstr(buf, "op_in");
    if (p)
    {
        sscanf(p, "op_in[%d] %s", &opnode_Num, opnode_name);
        sprintf(pdir_name, "%d.%s.xx.inputs",  opnode_Num, tensorName);
        return TRUE;
    }

    p = strstr(buf, "op_out");
    if (p)
    {
        assert(output_idx>=0);
        sscanf(p, "op_out[%d] %s", &opnode_Num, opnode_name);
        //sprintf(pdir_name, "%d.%s.xx.output%d",  opnode_Num, opnode_name,output_idx);
        sprintf(pdir_name, "%d.%s.xx.output%d",  opnode_Num, tensorName,output_idx);
        return TRUE;
    }

    return FALSE;
}

BOOL isHeadLine2(char *buf, int *buffer_size)
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

BOOL isEnd(const char *buf)
{
    if (strchr(buf, '}') == NULL)
    {
        return FALSE;
    }
    else
    {
        return TRUE;
    }
}

S32 getData(char *pBuffer, ST_DataType_e eDataType, int dataIdx)
{
    S32 s32Data = 0;
    if (eDataType == E_DATA_TYPE_UINT8)
    {
        s32Data = (S32)*((U8 *)pBuffer + dataIdx);
    }
    else if (eDataType == E_DATA_TYPE_INT8)
    {
        s32Data = (S32)*((S8 *)pBuffer + dataIdx);
    }
    else if (eDataType == E_DATA_TYPE_INT16)
    {
        s32Data = (S32)*((S16 *)pBuffer + dataIdx);
    }
    else
    {
        //need to do
        printf("unsuppported data type[%d]", eDataType);
        assert(0);
    }

    return s32Data;
}

#include <sys/stat.h>
#include <sys/types.h>
//int mkdir(const char *pathname, mode_t mode);


BOOL spliteFile(FILE *file)
{
    static char buffer0[1024];
    static char buffer1[1024];
    static char buffer2[512];
    static char buffer3[512];
    char *pBuffer = NULL;
    char dir_name[128];
    char file_name[128];
    char tensor_name[256];
    int ret = 0;
    char strNewFile[256];
    FILE *fData = NULL;
    int output_idx = -1;
    int buffer_size = 0;
    const char *pEnd = "\n};\n";
    ST_DataType_e eDataType = E_DATA_TYPE_MAX;
    memset(buffer3, 0, sizeof(buffer3));
    ret = fgets(buffer3, sizeof(buffer3), file);
    if (ret  == 0)
    {
        return FALSE;
    }
    sscanf(buffer3, "isFloat: %d", &bFloat);
    do {
        memset(dir_name, 0, sizeof(dir_name));
        memset(file_name, 0, sizeof(file_name));
        memset(buffer0, 0, sizeof(buffer0));
        if (buffer_size == 0)
        {
            ret = fgets(buffer0, sizeof(buffer0), file);
            if (ret  == 0)
            {
                break;
            }
            if (isHeadLine0(buffer0, file_name, &output_idx, &eDataType) == TRUE)
            {
                memset(buffer1, 0, sizeof(buffer1));
                ret = fgets(buffer1, sizeof(buffer1), file);
                if (ret  == 0)
                {
                    break;
                }
                if (isHeadLine1(buffer1, dir_name, output_idx, buffer0) == TRUE)
                {
                    //printf("mkdir %s\n", dir_name);
                    // makedir
                    if (access(dir_name, 0) != 0)
                    {
                        if (mkdir(dir_name, 0777) != 0)
                        {
                            perror("mkdir");
                            return FALSE;
                        }
                    }
                    else
                    {
                        printf("%s is exist.\n", dir_name);
                    }

                    // open new file
                    memset(strNewFile, 0, sizeof(strNewFile));
                    sprintf(strNewFile, "./%s/%s", dir_name, file_name);
                    //printf("create file name %s\n", strNewFile);
                    fData = fopen(strNewFile, "w");
                    if (fData == NULL)
                    {
                        perror("fopen");
                        return FALSE;
                    }
                    fputs(buffer3, fData);
                    fputs(buffer0, fData);
                    fputs(buffer1, fData);

                    memset(buffer2, 0, sizeof(buffer2));
                    ret = fgets(buffer2, sizeof(buffer2), file);

                    if (ret  == 0)
                    {
                        break;
                    }
                    if (isHeadLine2(buffer2,&buffer_size) == TRUE)
                    {
                    }
                }
                else
                {
                    perror("has headline0 but no headline1");
                    return FALSE;
                }
            }

            // write data to new file

            else if (isEnd(buffer0) == TRUE)
            {
                // close new file;
                //printf("close file: buffer: %s.\n", buffer);
                if (fData != NULL)
                {
                    fclose(fData);
                    fData = NULL;
                }
            }
        }
        else
        {
            pBuffer = (char*)calloc(1,buffer_size);
            ret = fread(pBuffer, buffer_size,1, file);
            char buf[512] = {0};
            BOOL bNeedWrite = FALSE;
            if (fData != NULL)
            {
                if (bFloat)
                {
                    float *f_Data = (float*)pBuffer;
                    for (int j = 0; j < buffer_size/4; j++)
                    {
                        if (j % 16 == 0)
                        {
                            sprintf(buf, "%s\n", buf);
                            fputs(buf, fData);
                            memset(buf, 0, sizeof(buf));
                            bNeedWrite = FALSE;
                        }
                        sprintf(buf, "%s%.6f, ", buf, *(f_Data+j));
                        bNeedWrite = TRUE;
                    }
                    if (bNeedWrite == TRUE)
                    {
                        sprintf(buf, "%s\n", buf);
                        fputs(buf, fData);
                        memset(buf, 0, sizeof(buf));
                        bNeedWrite = FALSE;
                    }
                }
                else
                {
                    int bytesWidth = 1;
                    if (eDataType == E_DATA_TYPE_INT16)
                    {
                        bytesWidth *= 2;
                    }

                    for (int j = 0; j < buffer_size / bytesWidth; j++)
                    {
                        if (j % 16 == 0)
                        {
                            sprintf(buf, "%s\n", buf);
                            fputs(buf, fData);
                            memset(buf, 0, sizeof(buf));
                            bNeedWrite = FALSE;
                        }
                        sprintf(buf, "%s%6d, ", buf, getData(pBuffer, eDataType ,j));
                        bNeedWrite = TRUE;
                    }
                    if (bNeedWrite == TRUE)
                    {
                        sprintf(buf, "%s\n", buf);
                        //write(fd, buf, strlen(buf));
                        fputs(buf, fData);
                        memset(buf, 0, sizeof(buf));
                        bNeedWrite = FALSE;
                    }
                }
                fputs(pEnd, fData);
            }
            free(pBuffer);
            buffer_size = 0;
        }
    } while (1);
    return TRUE;
}


int main(int argc, const char *argv[])
{
    FILE *file = NULL;
    char *p = NULL;
    char new_name[256] = {0};
    char file_name[256] = {0};

    if (argc < 2)
    {
        printf("%s <filename>\n", argv[0]);
        return 0;
    }
    strcpy(file_name,argv[1]);
    p = strtok(file_name, ".");
    sprintf(new_name, "%s.c",  p);
    file_out_name = new_name;
    file = fopen(argv[1], "rb");
    if (file != NULL)
    {
        // xxx
        spliteFile(file);
        fclose(file);
    }
    else
    {
        perror("fopen");
    }
    return 0;
}

