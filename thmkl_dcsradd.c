//#include "new_thmkl_spblas.h"
//#include "../include/thmkl.h"
#include"mmio_highlevel.h"
#include"mmio.h"
#include "omp.h"

void thmkl_dcsradd(const char *trans, const int *request, const int *sort,
                 const int *m, const int *n, double *a, int *ja, int *ia, const
                 double *beta, double *b, int *jb, int *ib, double *c, int *jc, int
                 *ic, const int *nzmax, int *info){
    if (*request==1)
    {
        int i,j,k;
        int *nnzr = (int *)malloc((*m)*sizeof(int));
       // memset(nnzr, 0, (*m)*sizeof(int));
       #pragma omp parallel for
        for (i=0;i<*m;i++)
        {
            j=ia[i]; 
            k=ib[i]; 
            nnzr[i]=0;
            while (j<ia[i+1]&&k<ib[i+1]) 
            {
                if (ja[j]<jb[k])
                {
                    nnzr[i]++;j++;
                }
                else if (ja[j]>jb[k])
                {
                   nnzr[i]++;k++;  
                }
                else{
                    nnzr[i]++;j++;k++;		
                }
            }
            if (k==ib[i+1]&&j<ia[i+1])
            {
                nnzr[i]+=ia[i+1]-j;
            }
            else if (k<ib[i+1]&&j==ia[i+1])
            { 
                nnzr[i]+=ib[i+1]-k;
            } 
            printf("nnz%d=%d\n",i,nnzr[i]);
        }
        
        for (i=0;i<(*m)+1;i++){
            ic[i+1]=ic[i]+nnzr[i];
        }

    }
    if (*request !=1)
    {
        int j,k;
        int count;
        #pragma omp parallel for
        for (int i=0;i<*m;i++)
        {
            j=ia[i];
            k=ib[i];
            count=ic[i];
        //	printf("count=%d\n",count);
            while (j<ia[i+1]&&k<ib[i+1]) 
            {
                if (ja[j]<jb[k])
                {
                    c[count]=a[j];jc[count]=ja[j]; j++; count++;
                }
                else if (ja[j]>jb[k])
                {
                    c[count]=b[k];jc[count]=jb[k]; k++; count++;
                }
                else{
                    c[count]=a[j]+b[k];jc[count]=ja[j];count++;
                    j++;k++;		
                }
            }	
            if (k==ib[i+1]&&j<ia[i+1])
            {
                while(j<ia[i+1])
                {
                    c[count]=a[j];jc[count]=ja[j]; 
                    count++;j++; 
                }
            }
            else if (k<ib[i+1]&&j==ia[i+1])
            {
                while(k<ib[i+1])
                {
                    c[count]=b[k];ic[count]=ib[k]; 
                    count++;k++; 
                }
            } 
        }
    }
}

int main(int argc, char ** argv)
{
//	printf("--------------------------------!!!!!!!!------------------------------------\n");
	int argi = 1;

    char  *filename1;
	char  *filename2;
    if(argc > argi)
    {
        filename1 = argv[argi];
        argi++;
    }
	//int row;
	int mA;
	int nA;
	int nnzA;
	int isSymmetricA;
	mmio_info(&mA, &nA, &nnzA, &isSymmetricA, filename1);
	printf("input matrix A: ( %i, %i ) nnz = %i\n", mA, nA, nnzA);

    int *csrRowPtrA;
    int *csrColIdxA;
    double *csrValA;
	
	csrRowPtrA=(int*)malloc((mA+1)*sizeof(int));
	csrColIdxA=(int*)malloc((nnzA)*sizeof(int));
	csrValA=(double*)malloc((nnzA)*sizeof(double));

	
	mmio_data(csrRowPtrA,csrColIdxA,csrValA, filename1);
	
	if(argc > argi)
    {
        filename2 = argv[argi];
        argi++;
    }
	int mB;
	int nB;
	int nnzB;
	int isSymmetricB;
	mmio_info(&mB, &nB, &nnzB, &isSymmetricB, filename2);
    int *csrRowPtrB;
    int *csrColIdxB;
    double *csrValB;
	csrRowPtrB=(int*)malloc((mB+1)*sizeof(int));
	csrColIdxB=(int*)malloc((nnzB)*sizeof(int));
	csrValB=(double*)malloc((nnzB)*sizeof(double));
	mmio_data(csrRowPtrB,csrColIdxB,csrValB, filename2);
	printf("input matrix B: ( %i, %i ) nnz = %i\n", mB, nB, nnzB);

	int request,sort;
	int *csrRowPtrC;
    int *csrColIdxC;
    double *csrValC;
	int nnzc;
	double beta;
	int info;
	request=1;

	csrRowPtrC=(int *)malloc((mA+1)*sizeof(int));
	memset(csrRowPtrC,0,(mA+1)*sizeof(int));

    struct timeval t1, t2;
	gettimeofday(&t1, NULL);
	thmkl_dcsradd("n",&request,&sort,&mA,&nA,csrValA,csrColIdxA,csrRowPtrA,&beta,
                     csrValB,csrColIdxB,csrRowPtrB,csrValC,csrColIdxC,
                     csrRowPtrC,&nnzc,&info);
                    
   
//	csrRowPtrC=(int*)malloc((rowA+1)*sizeof(int));
   
    nnzc=csrRowPtrC[mA];
    printf("nnzc=%d\n",nnzc);
	//nnzc=csrRowPtrC[mA];
	csrColIdxC=(int*)malloc((nnzc)*sizeof(int));
	csrValC=(double*)malloc((nnzc)*sizeof(double));

	memset(csrValC,0,(nnzc)*sizeof(double));
	memset(csrColIdxC,0,(nnzc)*sizeof(int));

//	request=2;
//	thmkl_dcsradd("n",&request,&sort,&mA,&nA,csrValA,csrColIdxA,csrRowPtrA,&beta,
//                     csrValB,csrColIdxB,csrRowPtrB,csrValC,csrColIdxC,
//                     csrRowPtrC,&nnzc,&info);

    gettimeofday(&t2, NULL);
	double time_parallel = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("dcsradd used %4.2f ms\n", time_parallel);

/*	for (int i=0;i<nnzc;i++)
	{
		printf("%f   ",csrValC[i]);
	}
	printf("\n");
	for (int i=0;i<nnzc;i++)
	{
		printf("%d   ",csrColIdxC[i]);
	}
	printf("\n");
	for (int i=0;i<mA+1;i++)
	{
		printf("%d   ",csrRowPtrC[i]);
	}
	printf("\n");
*/

}
