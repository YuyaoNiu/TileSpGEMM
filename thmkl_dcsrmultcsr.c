//#include "new_thmkl_spblas.h"
#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "mmio_highlevel.h"

void thmkl_dcsrmultcsr(const char *trans, const int *request, const int *sort, 
                       const int *m, const int *n, const int *k, double *a, int*ja, 
                       int *ia, double *b, int *jb, int *ib, double *c, int *jc,
                       int *ic, const int *nzmax, int *info)
{   
    if(*request == 1)
    {  

        #pragma omp parallel for
        for (int rid = 0; rid < (*m); rid++)
        {   int *d_dense_row_column_flag = (int *)malloc((*k)*sizeof(int));
            memset(d_dense_row_column_flag, 0, (*k)*sizeof(int));
            for (int rid_a = ia[rid]; rid_a < ia[rid+1]; rid_a++)
            {
                int rid_b = ja[rid_a];
                for (int cid_b = ib[rid_b]; cid_b < ib[rid_b+1]; cid_b++)
                {
                    d_dense_row_column_flag[jb[cid_b]] = 1;
                }
            }
            int nnzr = 0;
            for (int cid = 0; cid < *k; cid++)
            {   
                if (d_dense_row_column_flag[cid] == 1)
                {
                    nnzr++;
                }
            }
            ic[rid] = nnzr;
            free(d_dense_row_column_flag);
        }
        int new_val;
        int old_val;
        old_val=ic[0];
        ic[0]=0;
        for (int i = 1; i < (*m)+1; i++)
        {   
            new_val = ic[i];
            ic[i] = old_val + ic[i - 1];
            old_val = new_val;
        }
        printf("nnzc:%d\n",ic[(*m)]);
        
        *info = 1;  
    }
    if(*request == 0)
    {   
        #pragma omp parallel for
        for (int rid = 0; rid < (*m); rid++)
        {   int *d_dense_row_column_flag = (int *)malloc((*k)*sizeof(int));
            double *d_dense_row_value = (double *)malloc((*k)*sizeof(double));
            memset(d_dense_row_column_flag, 0, (*k)*sizeof(int));
            memset(d_dense_row_value, 0, (*k)*sizeof(double));
                
            for (int rid_a = ia[rid]; rid_a < ia[rid+1]; rid_a++)
            {
                int rid_b = ja[rid_a];
                double val_a = 0;
                val_a = a[rid_a];
                    
                for (int cid_b = ib[rid_b]; cid_b < ib[rid_b+1]; cid_b++)
                {
                    d_dense_row_column_flag[jb[cid_b]] = 1;
                    d_dense_row_value[jb[cid_b]] += val_a*b[cid_b];
                }
            }

            int nnzr = 0;
            // printf("test1\n");
            for (int cid = 0; cid < *k; cid++)
            {   
                if (d_dense_row_column_flag[cid] == 1)
                {   
                    jc[ic[rid] + nnzr] = cid;
                    c[ic[rid] + nnzr] = d_dense_row_value[cid];
                    nnzr++;
                }
            }
            free(d_dense_row_column_flag);
            free(d_dense_row_value);
        }
            
            *info=0;
        
    }
            
         
        
}
int main(int argc, char ** argv)
{   int request;
    request=1;
    int sort; 
    sort=1;
/*    int mA;
	int nA;
    int nnzA;
	int isSymmetricA;
	int* csrRowPtrA;
	int* csrColIdxA;
	double* csrValA;
    int mB;
	int nB;
    int nnzB;
	int isSymmetricB;

 	int* csrRowPtrB;
	int* csrColIdxB;
	double* csrValB; 
*/
    int mC;
    int nC;
	int nnzc;
	int* csrRowPtrC;
	int* csrColIdxC;
	double* csrValC;
    int nzmax;

    int info;
 /*   char filename1[]="DNN/neuron1024-1920/n1024-l1.tsv";
	mmio_info(&mA, &nA, &nnzA, &isSymmetricA, filename1);	
    csrRowPtrA=(int *)malloc((mA+1)*sizeof(int));
    csrColIdxA=(int *)malloc(nnzA*sizeof(int));
    csrValA=(double *)malloc(nnzA*sizeof(double));
    mmio_data(csrRowPtrA, csrColIdxA, csrValA, filename1);
    printf("input matrix A: ( %i, %i ) nnz = %i\n", mA, nA, nnzA);

    char filename2[]="DNN/neuron1024-1920/n1024-l1.tsv";
	mmio_info(&mB, &nB, &nnzB, &isSymmetricB, filename2);
    csrRowPtrB=(int *)malloc((mB+1)*sizeof(int));
    csrColIdxB=(int *)malloc(nnzB*sizeof(int));
    csrValB=(double *)malloc(nnzB*sizeof(double));
    mmio_data(csrRowPtrB,csrColIdxB,csrValB, filename2);
    printf("input matrix B: ( %i, %i ) nnz = %i\n", mB, nB, nnzB);
*/
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


    struct timeval t1, t2;
	gettimeofday(&t1, NULL);
    csrRowPtrC=(int *)malloc((mA+1)*sizeof(int));
    
    thmkl_dcsrmultcsr("n",&request,&sort,&mA,&nA,&nB,csrValA,csrColIdxA,csrRowPtrA,
                     csrValB,csrColIdxB,csrRowPtrB,csrValC,csrColIdxC,
                     csrRowPtrC,&nzmax,&info);
    nnzc=csrRowPtrC[mA];
    printf("nnzc %d\n",nnzc); 
    csrColIdxC=(int *)malloc(nnzc*sizeof(int));
    csrValC=(double *)malloc(nnzc*sizeof(double));
    printf("info:%d\n",info);

    request=0;
    thmkl_dcsrmultcsr("n",&request,&sort,&mA,&nA,&nB,csrValA,csrColIdxA,csrRowPtrA,
                     csrValB,csrColIdxB,csrRowPtrB,csrValC,csrColIdxC,
                     csrRowPtrC,&nzmax,&info);
    printf("info:%d\n",info);
    gettimeofday(&t2, NULL);
	double time_parallel = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("dcsrmulcsr used %4.2f ms\n", time_parallel);
    
    
}
