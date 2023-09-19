#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <iostream>
#include <sys/stat.h>
#include <dirent.h>
#include <fcntl.h>
#include <map>

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
static const char *file_in_name = "input";
static const char *file_out_name = "output";
BOOL bFloat = TRUE;
BOOL bSkipInput = FALSE;

typedef struct {
    char *buf;
    int size;
} BufferInfo_t;

typedef struct {
    std::map<int, BufferInfo_t> stInMap;
    BufferInfo_t Out;
} OP_InOutInfo_t;

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

int remove_dir(const char *dir)
{
    char cur_dir[] = ".";
    char up_dir[] = "..";
    char dir_name[128];
    DIR *dirp;
    struct dirent *dp;
    struct stat dir_stat;

    // 参数传递进来的目录不存在，直接返回
    if ( 0 != access(dir, F_OK) ) {
        return 0;
    }

    // 获取目录属性失败，返回错误
    if ( 0 > stat(dir, &dir_stat) ) {
        perror("get directory stat error");
        return -1;
    }

    if ( S_ISREG(dir_stat.st_mode) ) {  // 普通文件直接删除
        remove(dir);
    } else if ( S_ISDIR(dir_stat.st_mode) ) {   // 目录文件，递归删除目录中内容
        dirp = opendir(dir);
        while ( (dp=readdir(dirp)) != NULL ) {
            // 忽略 . 和 ..
            if ( (0 == strcmp(cur_dir, dp->d_name)) || (0 == strcmp(up_dir, dp->d_name)) ) {
                continue;
            }

            sprintf(dir_name, "%s/%s", dir, dp->d_name);
            remove_dir(dir_name);   // 递归调用
        }
        closedir(dirp);

        rmdir(dir);     // 删除空目录
    } else {
        perror("unknow file type!");
    }
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

BOOL isHeadLine0(char *buf, char *pfile_name, int *tensor_idx, ST_DataType_e *peDataType)
{
    char *p = NULL;
    char in_out_str[8];
    float scale = 0;
    S64 s64ZeroPoint = 0;
    char dataName[10];

    p = strchr(buf, '/');
    if (p == NULL)
        return FALSE;

    memset(in_out_str, 0, sizeof(in_out_str));
    memset(dataName, 0, sizeof(dataName));
    sscanf(p, "//%s %d s: %f z: %lld type: %s name", in_out_str, tensor_idx, &scale, &s64ZeroPoint, dataName);

    if (strcmp("in", in_out_str) == 0) {
        sprintf(pfile_name, "%s%d",  file_in_name, *tensor_idx);
    }
    else if (strcmp("out", in_out_str) == 0) {
        //sprintf(pfile_name, "%s%d",  file_out_name, file_num);
        sprintf(pfile_name, "%s",  file_out_name);
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

BOOL isHeadLine1(char *buf, char *pdir_name, int tensor_idx, char *buf0, BOOL *pBIn, int *pOpIndex, const char *prefix)
{
    char *p = NULL;
    //int opnode_Num = -1;
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
        *pBIn = TRUE;
        sscanf(p, "op_in[%d] %s", pOpIndex, opnode_name);
        sprintf(pdir_name, "%s/%d.%s.xx.inputs", prefix, *pOpIndex, tensorName);
        return TRUE;
    }

    p = strstr(buf, "op_out");
    if (p)
    {
        *pBIn = FALSE;
        assert(tensor_idx>=0);
        sscanf(p, "op_out[%d] %s", pOpIndex, opnode_name);
        //sprintf(pdir_name, "%d.%s.xx.output%d",  opnode_Num, opnode_name,output_idx);
        sprintf(pdir_name, "%s/%d.%s.xx.output%d", prefix, *pOpIndex, tensorName,tensor_idx);
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

void handleBuffer(BOOL bIn, int tensor_idx, OP_InOutInfo_t &stOpInfo, char *pBuffer, int buffer_size)
{
    if (bIn)
    {
        BufferInfo_t stBuf;
        stBuf.size = buffer_size;
        stBuf.buf = pBuffer;
        stOpInfo.stInMap.emplace(tensor_idx, stBuf);
    }
    else
    {
        assert(tensor_idx>=0);
        stOpInfo.Out.size = buffer_size;
        stOpInfo.Out.buf = pBuffer;
    }
}
#include <sys/stat.h>
#include <sys/types.h>
//int mkdir(const char *pathname, mode_t mode);

BOOL spliteFile(FILE *file, std::map<int, OP_InOutInfo_t> &bufMap, const char *prefix)
{
    char buffer0[1024];
    char buffer1[1024];
    char buffer2[512];
    char buffer3[512];
    char *pBuffer = NULL;
    char dir_name[128];
    char file_name[128];
    char tensor_name[256];
    int ret = 0;
    char strNewFile[256];
    FILE *fData = NULL;
    int tensor_idx = -1;
    int buffer_size = 0;
    const char *pEnd = "\n};\n";
    int opnode_Num = 0;
    BOOL bIn = TRUE;
    ST_DataType_e eDataType = E_DATA_TYPE_MAX;
    memset(buffer3, 0, sizeof(buffer3));

    if (access(prefix, 0) == 0)
    {
        remove_dir(prefix);
    }

    if (mkdir(prefix, 0777) != 0)
    {
        perror("mkdir");
        return FALSE;
    }

    if (fgets(buffer3, sizeof(buffer3), file)  == NULL)
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
            if (fgets(buffer0, sizeof(buffer0), file)  == NULL)
            {
                break;
            }
            if (isHeadLine0(buffer0, file_name, &tensor_idx, &eDataType) == TRUE)
            {
                memset(buffer1, 0, sizeof(buffer1));
                if (fgets(buffer1, sizeof(buffer1), file)  == NULL)
                {
                    break;
                }
                if (isHeadLine1(buffer1, dir_name, tensor_idx, buffer0, &bIn, &opnode_Num, prefix) == TRUE)
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
                    if (fgets(buffer2, sizeof(buffer2), file)  == NULL)
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
            auto it = bufMap.find(opnode_Num);
            if (it != bufMap.end())
            {
                handleBuffer(bIn, tensor_idx, it->second, pBuffer, buffer_size);
            }
            else
            {
                OP_InOutInfo_t stOpInfo;
                memset(&stOpInfo.Out, 0, sizeof(BufferInfo_t));
                stOpInfo.stInMap.clear();
                handleBuffer(bIn, tensor_idx, stOpInfo, pBuffer, buffer_size);
                bufMap.emplace(opnode_Num, stOpInfo);
            }
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
            //free(pBuffer);
            pBuffer = NULL;
            buffer_size = 0;
        }
    } while (1);
    return TRUE;
}

void destroyTwoMap(std::map<int, OP_InOutInfo_t> &goldenMap, std::map<int, OP_InOutInfo_t> &TBMap)
{
    for (auto &it : goldenMap)
    {
        free(it.second.Out.buf);
        for (auto &its : it.second.stInMap)
        {
            free(its.second.buf);
        }
    }

    for (auto &it : TBMap)
    {
        free(it.second.Out.buf);
        for (auto &its : it.second.stInMap)
        {
            free(its.second.buf);
        }
    }
}

BOOL compareTwoMap(std::map<int, OP_InOutInfo_t> &goldenMap, std::map<int, OP_InOutInfo_t> &TBMap)
{
    if (TBMap.size() != goldenMap.size())
    {
        printf("ERROR: golden and tile based have different op size, golden = %ld, TB = %ld \n", goldenMap.size(), TBMap.size());
        return FALSE;
    }

    for (auto &it : TBMap)
    {
        int op_idx = it.first;
        OP_InOutInfo_t &stGDOp = goldenMap[op_idx];
        OP_InOutInfo_t &stTBOp = TBMap[op_idx];
        if (stTBOp.stInMap.size() != stGDOp.stInMap.size())
        {
            printf("ERROR: op[%d] input tensor size are not same!!\n", op_idx);
            assert(0);
        }
        if(bSkipInput == FALSE)
        {
            for (auto &its : TBMap[op_idx].stInMap)
            {
                int tensor_idx = its.first;
                BufferInfo_t &stTBBuffer = stTBOp.stInMap[tensor_idx];
                BufferInfo_t &stGDBuffer = stGDOp.stInMap[tensor_idx];
                if (stTBBuffer.size != stGDBuffer.size ||
                    memcmp(stTBBuffer.buf, stGDBuffer.buf, stGDBuffer.size) != 0)
                {
                    printf("ERROR: op[%d]-->input tensor[%d] are not same!!\n", op_idx, tensor_idx);
                    assert(0);
                }
            }
        }

        if (stTBOp.Out.size != stGDOp.Out.size ||
            memcmp(stTBOp.Out.buf, stGDOp.Out.buf, stGDOp.Out.size) != 0)
        {
            printf("ERROR: op[%d] output are not same!!\n", op_idx);
            assert(0);
        }

    }

    return TRUE;
}

void showUsage()
{
    printf("Param1: golden.bin\n");
    printf("Param2: tile_based.bin\n");
    printf("Example, You could try command like:\n");
    printf("~/$(Project_PATH)/Tool/DumpDebug/verify_tile_based ./golden.bin ./tile_based.bin\n");
    printf("if Success is printed in Console, tile based bin verify OK\n");
    printf("if Error is printed in Console, you can get op_index in Console\n");
    printf("you can check data by op_index in documents which are named as golden and TB\n");
}

int main(int argc, const char *argv[])
{
    FILE *golden_file = NULL;
    FILE *TB_file = NULL;
    std::map<int, OP_InOutInfo_t> goldenMap;
    std::map<int, OP_InOutInfo_t> TBMap;

    if (strcmp(argv[1], "-h") == 0 ||
        strcmp(argv[1], "--help") == 0)
    {
        showUsage();
        return 0;
    }

    if (argc != 4)
    {
        printf("need two file:1. golden.bin; 2. tile based.bin 3.skipinput\n");
        return 0;
    }

    golden_file = fopen(argv[1], "rb");
    if (golden_file != NULL)
    {
        spliteFile(golden_file, goldenMap, "./golden");
        fclose(golden_file);
    }
    else
    {
        perror("fopen");
    }

    TB_file = fopen(argv[2], "rb");
    if (TB_file != NULL)
    {
        spliteFile(TB_file, TBMap, "./TB");
        fclose(TB_file);
    }
    else
    {
        perror("fopen");
    }

    if(!strcmp(argv[3], "skipinput"))
    {
        bSkipInput = TRUE;
    }
    if (TRUE == compareTwoMap(goldenMap, TBMap))
    {
        printf("Success: TB bin and golden bin are same!!!\n");
    }
    destroyTwoMap(goldenMap, TBMap);


    return 0;
}

