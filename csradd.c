#include"common.h"
#include"mmio_highlevel.h"
#include"mmio.h"
#include "omp.h"


int main(int argc, char ** argv)
{
	printf("--------------------------------!!!!!!!!------------------------------------\n");
	int argi = 1;

    char  *filename1;
	char  *filename2;
    if(argc > argi)
    {
        filename1 = argv[argi];
        argi++;
    }
	int row;
	int rowA;
	int colA;
	int nnzA;
	int isSymmetricA;
	mmio_info(&rowA, &colA, &nnzA, &isSymmetricA, filename1);
	printf("rowA=%d   ",rowA);
	printf("colA=%d   ",colA);
	printf("nnzA=%d\n",nnzA);

    int *csrRowPtrA;
    int *csrColIdxA;
    double *csrValA;
	
	csrRowPtrA=(int*)malloc((rowA+1)*sizeof(int));
	csrColIdxA=(int*)malloc((nnzA)*sizeof(int));
	csrValA=(double*)malloc((nnzA)*sizeof(double));

	
	mmio_data(csrRowPtrA,csrColIdxA,csrValA, filename1);
	
	if(argc > argi)
    {
        filename2 = argv[argi];
        argi++;
    }
	int rowB;
	int colB;
	int nnzB;
	int isSymmetricB;
	mmio_info(&rowB, &colB, &nnzB, &isSymmetricB, filename2);
    int *csrRowPtrB;
    int *csrColIdxB;
    double *csrValB;
	csrRowPtrB=(int*)malloc((rowB+1)*sizeof(int));
	csrColIdxB=(int*)malloc((nnzB)*sizeof(int));
	csrValB=(double*)malloc((nnzB)*sizeof(double));
	mmio_data(csrRowPtrB,csrColIdxB,csrValB, filename2);
	printf("rowB=%d   ",rowB);
	printf("colB=%d   ",colB);
	printf("nnzB=%d\n",nnzB);

    int j=0,k=0,nnzC=0;

	int *csrRowPtrC;
	csrRowPtrC=(int*)malloc((rowA+1)*sizeof(int));
	memset(csrRowPtrC,0,(rowA+1)* sizeof(int));
    for (int i=0;i<rowA;i++)
	{
		while (j<csrRowPtrA[i+1]&&k<csrRowPtrB[i+1]) 
		{
			if (csrColIdxA[j]<csrColIdxB[k])
			{
				nnzC++;j++;
			}
			else if (csrColIdxA[j]>csrColIdxB[k])
			{
				nnzC++;k++;  
			}
			else{
			//	if (csrValA[j]+csrValB[k]!=0)
			//	{
					nnzC++;
			//	}
				j++;k++;		
			}
		}	

		if (k==csrRowPtrB[i+1]&&j<csrRowPtrA[i+1])
		{
			nnzC+=csrRowPtrA[i+1]-j;
		}
		else if (k<csrRowPtrB[i+1]&&j==csrRowPtrA[i+1])
		{
			nnzC+=csrRowPtrB[i+1]-k;
		} 
		csrRowPtrC[i+1]=nnzC;
	}
   printf("nnzC=%d\n",nnzC);

//    int *csrRowPtrC;
    int *csrColIdxC;
    double *csrValC;
//	csrRowPtrC=(int*)malloc((rowA+1)*sizeof(int));
	csrColIdxC=(int*)malloc((nnzC)*sizeof(int));
	csrValC=(double*)malloc((nnzC)*sizeof(double));

	memset(csrValC,0,(nnzC)*sizeof(double));
	memset(csrColIdxC,0,(nnzC)*sizeof(int));
//	csrRowPtrC[0]=0;
	//j=0;k=0;
	int count;
#pragma omp parallel for
	for (int i=0;i<rowA;i++)
	{
	//	printf("the %dth row\n",i);
		j=csrRowPtrA[i];
		k=csrRowPtrB[i];
		count=csrRowPtrC[i];
		//printf("count=%d\n",count);
		while (j<csrRowPtrA[i+1]&&k<csrRowPtrB[i+1]) 
		{
			if (csrColIdxA[j]<csrColIdxB[k])
			{
				csrValC[count]=csrValA[j];csrColIdxC[count]=csrColIdxA[j]; j++; count++;
			}
			else if (csrColIdxA[j]>csrColIdxB[k])
			{
				csrValC[count]=csrValB[k];csrColIdxC[count]=csrColIdxB[k]; k++; count++;
			}
			else{
			//	if (csrValA[j]+csrValB[k]!=0)
			//	{
					csrValC[count]=csrValA[j]+csrValB[k];csrColIdxC[count]=csrColIdxA[j];count++;
			//	}
				j++;k++;		
			}
		}	

		if (k==csrRowPtrB[i+1]&&j<csrRowPtrA[i+1])
		{
			while(j<csrRowPtrA[i+1])
			{
				csrValC[count]=csrValA[j];csrColIdxC[count]=csrColIdxA[j]; 
				count++;j++; 
			}
		}
		else if (k<csrRowPtrB[i+1]&&j==csrRowPtrA[i+1])
		{
			while(k<csrRowPtrB[i+1])
			{
				csrValC[count]=csrValB[k];csrColIdxC[count]=csrColIdxB[k]; 
				count++;k++; 
			}
		} 
	}

/*	for (int i=0;i<nnzC;i++)
	{
		printf("%f   ",csrValC[i]);
	}
	printf("\n");
	for (int i=0;i<nnzC;i++)
	{
		printf("%d   ",csrColIdxC[i]);
	}
	printf("\n");
	for (int i=0;i<rowA+1;i++)
	{
		printf("%d   ",csrRowPtrC[i]);
	}
	printf("\n");
*/

    return 0;
}

