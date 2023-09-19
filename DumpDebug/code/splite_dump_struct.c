#include <stdio.h>
#include <string.h>
typedef int BOOL;

#define TRUE  (1)
#define FALSE (0)

BOOL isHead(char *buf, char *pname)
{
	char *p = NULL;
	char *pNum = NULL;
	char *pNumEnd = NULL;
	char *pOpName = NULL;
	char *pOpNameEnd = NULL;
	char strNum[16];
	char strName[64];
	p = strchr(buf, '{');
	if (p != NULL)
	{
		// find [
		p = strchr(buf, '[');
		if (p != NULL)
		{
			pNum = p + 1;
		// find ']'
			pNumEnd = strchr(pNum, ']');
			if (pNumEnd != NULL)
			{
				memset(strNum, 0, sizeof(strNum));
				strncpy(strNum, pNum, (pNumEnd - pNum));
				pOpName = pNumEnd + 2;
				//printf("pOpName: %s\n", pOpName);
				pOpNameEnd = strchr(pOpName, ' ');
				if (pOpNameEnd != NULL)
				{
					memset(strName, 0, sizeof(strName));
					strncpy(strName, pOpName, (pOpNameEnd - pOpName));
					sprintf(pname, "%s.%s.xx",  strNum,  strName);
					//sprintf(pname, "%s.%s",  pname, strNum);
					//printf("pname: %s.\n", pname);
				}
			}
		}
		return TRUE;
	}
	else
	{
		return FALSE;
	}
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

#include <sys/stat.h>
#include <sys/types.h>
//int mkdir(const char *pathname, mode_t mode);

#define DESC_START "//out 0 s:"
#define DESC_TYPE  "type: "
#define DESC_NAME  "name: "
const char * getPattern( const char *pStr, const char *pstrPattern, char cEnd, char *pBuf)
{
	char *p = NULL;
	char *p1 = NULL;
	p = strstr(pStr, pstrPattern);
	if (p == NULL)
	{
		return p;	
	}

	p += strlen(pstrPattern);
	
	if (p != NULL && pBuf != NULL)
	{
		//printf("p: %s\n", p);
		p1 = strchr(p, cEnd);
		//printf("p1: %s\n", p1);
		if (p1 != NULL)
			strncpy(pBuf, p, (p1 - p));
	}

	return p;
}

const char * getEndPattern( const char *pStr, const char *pstrPattern, char *pBuf)
{
	char *p = NULL;
	p = strstr(pStr, pstrPattern);
	if (p == NULL)
	{
		return p;	
	}

	if (p != NULL && pBuf != NULL)
	{
		strncpy(pBuf, pStr, (p - pStr));
	}

	return p;
}

const char *strRepalceChr(char *pStr, char cSrc, char cDest)
{
	char *p = pStr;
	do {
		p = strchr(p, cSrc);
		if (p == NULL)
		{
			break;
		}
		*p = cDest;
		p++;
	} while (*p != '\0');

	return pStr;
}
BOOL spliteFile(FILE *file, FILE *fOutCommon, FILE *fOutTop, const char *pfileName)
{
	static char buffer[512];
	char head[32];
	int ret = 0;
	char strNewFile[64];
	FILE *fData = NULL;
	char strType[16];
	static char strName[128];
	static char strDataName[128];
	static char strStruct[128];
	char *p = NULL;

	do {
		memset(buffer, 0, sizeof(buffer));
		memset(head, 0, sizeof(head));
		ret = fgets(buffer, sizeof(buffer), file);
		if (ret < 0)
		{
			perror("fgets");
		}
		if (ret  == 0)
		{
			break;
		}

		p = getPattern(buffer, DESC_START, ' ', NULL);
		if (p != NULL)
		{
			memset(strType, 0, sizeof(strType));	
			memset(strName, 0, sizeof(strName));	
			getPattern(buffer, DESC_TYPE, ' ', strType);	
			getPattern(buffer, DESC_NAME, '\n', strName);	
		}

		if (isHead(buffer, head) == TRUE)
		{
			// open new file
			memset(strDataName, 0, sizeof(strDataName));
			if (getEndPattern(buffer, " = ", strDataName) != NULL)
			{
				strRepalceChr(strDataName, '[', '_');	
				strRepalceChr(strDataName, ']', '_');	
				strRepalceChr(strDataName, ' ', '_');	
			}
			memset(strNewFile, 0, sizeof(strNewFile));
			//sprintf(strNewFile, "./%s/%s", head, pfileName);
			sprintf(strNewFile, "./%s/%s.c", "./force_data/", strDataName);
			fData = fopen(strNewFile, "w");
			if (fData == NULL)
			{
				printf("%s()@line %d strNewFile: %s.\n", __func__, __LINE__, strNewFile);
				perror("fopen");
				return FALSE;
			}
			//sprintf(strStruct, "const char *pTensorName = \"%s\";\n", strName);
			//fputs(strStruct, fData);
			sprintf(strStruct, "#include \"%s.c\"\n", strDataName);
			fputs(strStruct, fOutTop);


			sprintf(strStruct, "    DEF_FORCE_DATA(%s, \"%s\"),\n", strDataName, strName);
			fputs(strStruct, fOutCommon);
			
			sprintf(strStruct, "%s %s[] = {\n", strType, strDataName);
			fputs(strStruct, fData);
		}
		else if (fData != NULL)
		{
			fputs(buffer, fData);
		}

		// write data to new file

		if (isEnd(buffer) == TRUE)
		{
			// close new file;
			//printf("close file: buffer: %s.\n", buffer);
			if (fData != NULL)
			{
				fclose(fData);
				fData = NULL;
			}
		}

	} while (1);

	return TRUE;
}

int main(int argc, const char *argv[])
{
	FILE * file = NULL;
	FILE * fCommonTop = NULL;
	FILE * fCommonBottom = NULL;
	static char buf[256];
	const char * head = "./force_data/";
	const char *pstrForceTop="./force_data_top.c";
	const char *pstrForceBottom="./force_data_bottom.c";

	if (argc < 2)
	{
		printf("%s <filename>\n", argv[0]);
		return 0;
	}

	//printf("mkdir %s\n", head);
	// makedir
	if (access(head, 0) != 0)
	{
		if (mkdir(head, 0777) != 0)
		{
			perror("mkdir");
			return FALSE;
		}
	}
	else
	{
		printf("%s is exist.\n", head);
	}


	file = fopen(argv[1], "r");
	if (file != NULL)
	{
		// xxx
		fCommonTop = fopen(pstrForceTop, "w");
		if (fCommonTop != NULL)
		{
			fCommonBottom = fopen(pstrForceBottom, "w");
			if (fCommonBottom != NULL)
			{
				sprintf(buf, "typedef struct {\n"
				"	void *pData;\n"
				"	const char * pstrName;\n"
				"} ST_ForceData_t;\n\n"
				"#define DEF_FORCE_DATA(data, str) .{ .pData = (data), .pstrName = (str), }\n\n");
				fputs(buf, fCommonTop);
				sprintf(buf, "static ST_ForceData_t _gaForceData[] = {\n");
				fputs(buf, fCommonBottom);
				spliteFile(file, fCommonBottom, fCommonTop, argv[1]);
				sprintf(buf, "\n};\n");
				fputs(buf, fCommonBottom);
				fclose(fCommonBottom);
				printf("Create:\n");
				printf("%s\n", head);
				printf("%s\n", pstrForceTop);
				printf("%s\n", pstrForceBottom);
			}
			fclose(fCommonTop);
		}
		fclose(file);
	}
	else
	{
		printf("%s()@line %d.\n", __func__, __LINE__);
		perror("fopen");
	}
	return 0;
}
