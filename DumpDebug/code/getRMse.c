#include <stdio.h>
#include <math.h>
#include "force_data.c"
#include "force_data2.c"
#define max(a,b)    (((a) > (b)) ? (a) : (b))
#define min(a,b)    (((a) < (b)) ? (a) : (b))

int main(int argc, const char *argv[])
{
	double fmin = 0;
	double fmax = 0;
	double *pData = op_out_force_data;
	double *pData2 = op_out_force_data2;
	int amax = 0;
	int index = 0;
	int i = 0;


	amax = sizeof(op_out_force_data)/sizeof(op_out_force_data[0]);
	
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
	for (i = 0; i < sizeof(op_out_force_data)/sizeof(op_out_force_data[0]); i++)
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
	printf("MSE:\t%f\tCOS:\t%f\tRMSE:\t%f\n", diffSum, cos, rdiffSum);
	
	return 0;
}

