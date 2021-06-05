#include"common.h"
#include"mmio_highlevel.h"
//#include"mmio.h"
#include"utils.h"

#include "ccsb_spmv.h"

typedef struct 
{
	MAT_VAL_TYPE *value;
	int *columnindex;
	MAT_PTR_TYPE *rowpointer;
}SMatrix;




void DivideSub(SMatrix A,int start,int over,SMatrix submatrix[],int sub,int nnznum[],int cbnum)
{
	int i,j,k;
	j=A.rowpointer[start];
	int *num;
	num=(int*)malloc((cbnum)*sizeof(int));
	//count=(int*)malloc((SubNum)*sizeof(int));
	memset(num,0,(cbnum)*sizeof(int));
	
	int temp=1;
	for (i=start;i<over;i++)
	{
		while(j<A.rowpointer[i+1]){
			for (k=0;k<cbnum;k++){
				if (nnznum[sub+k]!=0){
					if (k==cbnum-1){
						num[k]++;
						submatrix[sub+k].value[num[k]-1]=A.value[j];
						submatrix[sub+k].columnindex[num[k]-1]=A.columnindex[j]-k*BLOCK_SIZE;
					}
					else if (A.columnindex[j]>=k*BLOCK_SIZE&&A.columnindex[j]<(k+1)*BLOCK_SIZE){
						num[k]++;
						submatrix[sub+k].value[num[k]-1]=A.value[j];
						submatrix[sub+k].columnindex[num[k]-1]=A.columnindex[j]-k*BLOCK_SIZE;
						break;
					}
				}
			}
			j++;
		}
		for (int p=0;p<cbnum;p++){
			if (nnznum[sub+p]!=0)
				submatrix[sub+p].rowpointer[temp]=num[p];
		}      
		temp++;
		//printf("%d\n",num[p]);
	}
	free(num);
}
void nnzNum(SMatrix A,int nnz[],int start,int over,int i,int cbnum)
{
	//printf("start=%d,over=%d\n",start,over);
	//printf("Subcol=%d\n",SubCol);

	int j;
	for (j=A.rowpointer[start];j<A.rowpointer[over];j++){
		for (int k=0;k<cbnum-1;k++){
			if (A.columnindex[j]>=k*BLOCK_SIZE&&A.columnindex[j]<(k+1)*BLOCK_SIZE){
				nnz[i+k]++;
				break;
			}
		}
	} 
	int m=0;
	for (int p=i;p<i+cbnum-1;p++)
		m+= nnz[p];
		

	nnz[i+cbnum-1]=A.rowpointer[over]-A.rowpointer[start]-m;
}

int main(int argc, char ** argv)
{

	if (argc < 2)
    {
        printf("Run the code by './test matrix.mtx'.\n");
        return 0;
    }
	
    printf("--------------------------------!!!!!!!!------------------------------------\n");

 	struct timeval t1, t2;
	int rowA;
	int colA;
	MAT_PTR_TYPE nnzA;
	int isSymmetricA;
	SMatrix matrixA;


	char  *filename;
    filename = argv[1];
    printf("MAT: -------------- %s --------------\n", filename);

    // load mtx A data to the csr format
    gettimeofday(&t1, NULL);
    mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA, &matrixA.rowpointer, &matrixA.columnindex, &matrixA.value, filename);
    gettimeofday(&t2, NULL);
    double time_loadmat  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("input matrix A: ( %i, %i ) nnz = %i\n loadfile time    = %4.5f sec\n", rowA, colA, nnzA, time_loadmat/1000.0);

	for (int i = 0; i < nnzA; i++)
	    matrixA.value[i] = i % 10;

    if (rowA != colA)
    {
        printf("This code only computes square matrices.\n Exit.\n");
        return 0;
    }
	rowA=(rowA/BLOCK_SIZE) * BLOCK_SIZE;
	nnzA=matrixA.rowpointer[rowA];
//	MAT_PTR_TYPE *cscColPtrA = (MAT_PTR_TYPE *)malloc((colA+1) * sizeof(MAT_PTR_TYPE));
 //   int *cscRowIdxA = (int *)malloc(nnzA   * sizeof(int));
 //   MAT_VAL_TYPE *cscValA    = (MAT_VAL_TYPE *)malloc(nnzA  * sizeof(MAT_VAL_TYPE));


	 // transpose A from csr to csc
 //   matrix_transposition(rowA, colA, nnzA, matrixA.rowpointer, matrixA.columnindex, matrixA.value,cscRowIdxA, cscColPtrA, cscValA);

/*	SMatrix matrixB;
	int rowB=colA;
	int colB=rowA;

	matrixB.rowpointer = cscColPtrA;
    matrixB.columnindex = cscRowIdxA;
    matrixB.value    = cscValA;
*/
    if (BLOCK_SIZE>rowA){
		printf("Error!\n");
		return 0;
	}


    int rbnum=0;
    int cbnum=0;
    if (rowA%BLOCK_SIZE==0)
        rbnum=rowA/BLOCK_SIZE;
    else
        rbnum=(rowA/BLOCK_SIZE)+1;
    
    if (colA%BLOCK_SIZE==0)
        cbnum=colA/BLOCK_SIZE;
    else
        cbnum=(colA/BLOCK_SIZE)+1;
    
	SMatrix *submatrixA;
//	SMatrix *submatrixB;
//	SMatrix *submatrixC;
	submatrixA=(SMatrix*)malloc((rbnum*cbnum)*sizeof(SMatrix));
    
//	submatrixB=(SMatrix*)malloc((rbnum*cbnum)*sizeof(SMatrix));

//	submatrixC=(SMatrix*)malloc((rbnum*cbnum)*sizeof(SMatrix));

    int *nnzAnum,*nnzBnum;
	nnzAnum=(int*)malloc((rbnum*cbnum)*sizeof(int));
//	nnzBnum=(int*)malloc((rbnum*cbnum)*sizeof(int));
	//nnzCnum=(int*)malloc((SubNum*SubNum)*sizeof(int));
   memset(nnzAnum,0,(rbnum*cbnum)*sizeof(int));
//   memset(nnzBnum,0,(rbnum*cbnum)*sizeof(int));

	
//calculate nnz in each block

	#pragma omp parallel for

	for (int i=0;i<rbnum-1;i++)
	{
		nnzNum(matrixA,nnzAnum,i*BLOCK_SIZE,(i+1)*BLOCK_SIZE,i*cbnum,cbnum);
	}
	nnzNum(matrixA,nnzAnum,(rbnum-1)*BLOCK_SIZE,rowA,(rbnum-1)*cbnum,cbnum);

/*	for (int i=0;i<cbnum-1;i++)
	{
		nnzNum(matrixB,nnzBnum,i*BLOCK_SIZE,(i+1)*BLOCK_SIZE,i*cbnum,rbnum);
	}
	nnzNum(matrixB,nnzBnum,(cbnum-1)*BLOCK_SIZE,rowB,(cbnum-1)*cbnum,rbnum);
*/
	int nnzbl=0;

	for (int i=0;i<rbnum*cbnum;i++)   //calculate number of non-zero blocks
	{
		if (nnzAnum[i]!=0)
		{
			nnzbl++;
		}
	}

 /*   for (int i=0;i<rbnum;i++)
    {
		for(int j=0;j<cbnum;j++)
		{
			if (nnzAnum[i*cbnum+j]!=0)
			 	printf("%d   ",nnzAnum[i*cbnum+j]);
		}
		printf("!!!!!!!!!!!!!\n");
    }
*/	

/*	for (int i=0;i<SubNum*SubNum;i++)
	{
		writeresults("blocksize.csv", filename, rowA, colA,nnzA,i,nnzAnum[i]);
	}
*/

	for (int i=0;i<rbnum*cbnum;i++)
	{
		int rowlength;
		if (nnzAnum[i]!=0)
		{
			submatrixA[i].value=(MAT_VAL_TYPE*)malloc((nnzAnum[i])*sizeof(MAT_VAL_TYPE));
			submatrixA[i].columnindex=(int*)malloc((nnzAnum[i])*sizeof(int));
			if (i<cbnum*(rbnum-1)){
				rowlength=BLOCK_SIZE;
			}
			else{
				rowlength=rowA-(rbnum-1)*BLOCK_SIZE;
			}
			submatrixA[i].rowpointer=(MAT_PTR_TYPE *)malloc((rowlength+1)*sizeof(MAT_PTR_TYPE));
			memset(submatrixA[i].rowpointer,0,(rowlength+1)*sizeof(MAT_PTR_TYPE));
		}
	/*	if (nnzBnum[i]!=0)
		{
			submatrixB[i].value=(MAT_VAL_TYPE*)malloc((nnzBnum[i])*sizeof(MAT_VAL_TYPE));
			submatrixB[i].columnindex=(int*)malloc((nnzBnum[i])*sizeof(int));

			if (i<rbnum*(cbnum-1)){
				rowlength=BLOCK_SIZE;
			}
			else{
				rowlength=rowB-(cbnum-1)*BLOCK_SIZE;
			}
			submatrixB[i].rowpointer=(MAT_PTR_TYPE*)malloc((rowlength+1)*sizeof(MAT_PTR_TYPE));
			memset(submatrixB[i].rowpointer,0,(rowlength+1)*sizeof(MAT_PTR_TYPE));
		}
	*/
	}

	MAT_PTR_TYPE *rowblock_ptr;    //block rowpointer of A
	int *columnid;                // block columnindex of A
	int *nnzb_A;
	int colid=0;
	rowblock_ptr=(MAT_PTR_TYPE *)malloc((rbnum+1)*sizeof(MAT_PTR_TYPE));
	columnid=(int *)malloc(nnzbl*sizeof(int));
	nnzb_A=(int *)malloc((nnzbl+1)*sizeof(int));
	memset(rowblock_ptr,0,(rbnum+1)*sizeof(MAT_PTR_TYPE));
	for (int i=0;i<rbnum;i++)
	{
		for (int j=0;j<cbnum;j++)
		{
			if (nnzAnum[i*cbnum+j]!=0)
			{
				columnid[colid]=j;
				nnzb_A[colid]=nnzAnum[i*cbnum+j];
				rowblock_ptr[i+1]++;
				colid++;
			}
		}
	}
	for (int i=1;i<rbnum+1;i++)
	{
		rowblock_ptr[i]+=rowblock_ptr[i-1];
	}
	/*	for (int i=0;i<nnzbl;i++)
		{
			printf("%d    ",nnzb[i]);
		}
	*/
	exclusive_scan(nnzb_A,nnzbl+1);

/*	MAT_PTR_TYPE *colblock_ptr;    //block columnpointer of A
	int *rowid;           // block rowindex of B
	int *nnzb_B;
	int rid=0;
	colblock_ptr=(MAT_PTR_TYPE *)malloc((rbnum+1)*sizeof(MAT_PTR_TYPE));
	rowid=(int *)malloc(nnzbl*sizeof(int));
	nnzb_B=(int *)malloc(nnzbl*sizeof(int));
	memset(colblock_ptr,0,(rbnum+1)*sizeof(MAT_PTR_TYPE));

	for (int i=0;i<rbnum;i++)
	{
		for (int j=0;j<cbnum;j++)
		{
			if (nnzBnum[i%rbnum+j*rbnum]!=0)
			{
				rowid[rid]=j;
				nnzb_B[rid]=nnzBnum[i%rbnum+j*rbnum];
				colblock_ptr[i+1]++;
				rid++;
			}
		}
	}
	for (int i=1;i<rbnum+1;i++)
	{
		colblock_ptr[i]+=colblock_ptr[i-1];
	}
*/
/*	for (int i=0;i<cbnum+1;i++)
	{
		printf("%d    ",colblock_ptr[i]);
	}
	printf("\n");
*/
	for (int i=0;i<rbnum-1;i++)
	{
		DivideSub(matrixA,i*BLOCK_SIZE,(i+1)*BLOCK_SIZE,submatrixA,i*cbnum,nnzAnum,cbnum);
		
	}
	DivideSub(matrixA,(rbnum-1)*BLOCK_SIZE,rowA,submatrixA,(rbnum-1)*cbnum,nnzAnum,cbnum);

/*	for (int i=0;i<cbnum-1;i++)
	{
		DivideSub(matrixB,i*BLOCK_SIZE,(i+1)*BLOCK_SIZE,submatrixB,i*cbnum,nnzBnum,rbnum);
	}
	DivideSub(matrixB,(cbnum-1)*BLOCK_SIZE,rowB,submatrixB,(cbnum-1)*cbnum,nnzBnum,rbnum);
*/

/*	for (int i=0;i<rbnum*cbnum;i++)
	{
		for (int j=0;j<nnzBnum[i];j++)
		{
			printf("%d     ",submatrixB[i].columnindex[j]);
		}
		printf("\n");
	}
*/


	MAT_VAL_TYPE *BlockA_Val=(MAT_VAL_TYPE*)malloc((nnzA)*sizeof(MAT_VAL_TYPE));
	char  *BlockA_Col=(char*)malloc((nnzA)*sizeof(char));
	char *BlockA_Ptr;
	int vAid=0;
	int pAid=0;
	int ptrA_length=0;

	//calculate BlockA_Ptr length
	for (int i=0;i<rbnum;i++)
	{
		int rowlength;
		for (int j=rowblock_ptr[i];j<rowblock_ptr[i+1];j++)
		{
			int block=i*cbnum+columnid[j];
			if (block<(rbnum-1)*cbnum)
			{
				rowlength=BLOCK_SIZE;
			}
			else{
				rowlength=rowA-(rbnum-1)*BLOCK_SIZE;
			}
			ptrA_length+=(rowlength);
		}
	}
//	printf("ptrA length=%d\n",ptrA_length);
	BlockA_Ptr=(char*)malloc((ptrA_length)*sizeof(char));

	for (int i=0;i<rbnum;i++)
	{
		int rowlength;
		for (int j=rowblock_ptr[i];j<rowblock_ptr[i+1];j++)
		{
			int block=i*cbnum+columnid[j];
			if (block<(rbnum-1)*cbnum)
			{
				rowlength=BLOCK_SIZE;
			}
			else{
				rowlength=rowA-(rbnum-1)*BLOCK_SIZE;
			}
			for (int k=0;k<submatrixA[block].rowpointer[rowlength];k++)
			{
				BlockA_Val[vAid]=submatrixA[block].value[k];
				BlockA_Col[vAid]=submatrixA[block].columnindex[k];
				vAid++;
			}
			for (int jid=0;jid<rowlength;jid++)
			{
				BlockA_Ptr[pAid]=submatrixA[block].rowpointer[jid];
				pAid++;
			}
		}
	}

	

/*	for (int i=0;i<nnzA;i++)
	{
		printf("%f     ",BlockA_Val[i]);
	}
	printf("\n");
	for (int i=0;i<nnzA;i++)
	{
		printf("%d     ",BlockA_Col[i]);
	}
	printf("\n");
	for (int i=0;i<ptrA_length;i++)
	{
		printf("%d     ",BlockA_Ptr[i]);
	}
	printf("\n");
*/


/*	MAT_VAL_TYPE *BlockB_Val=(MAT_VAL_TYPE*)malloc((nnzA)*sizeof(MAT_VAL_TYPE));
	char  *BlockB_Col=(char*)malloc((nnzA)*sizeof(char));
	char *BlockB_Ptr;
	int vBid=0;
	int pBid=0;
	int ptrB_length=0;
	for (int i=0;i<rbnum;i++)
	{
		int rowlength;
		for (int j=colblock_ptr[i];j<colblock_ptr[i+1];j++)
		{
			int block=i%rbnum+rowid[j]*rbnum;
			if (block<(cbnum-1)*rbnum)
			{
				rowlength=BLOCK_SIZE;
			}
			else{
				rowlength=rowB-(cbnum-1)*BLOCK_SIZE;
			}
			ptrB_length+=(rowlength);
		}
	}
	printf("ptrB length=%d\n",ptrB_length);
	BlockB_Ptr=(char*)malloc((ptrB_length)*sizeof(char));

	for (int i=0;i<rbnum;i++)
	{
		int rowlength;
		for (int j=colblock_ptr[i];j<colblock_ptr[i+1];j++)
		{
			int block=i%rbnum+rowid[j]*rbnum;
			if (block<(cbnum-1)*rbnum)
			{
				rowlength=BLOCK_SIZE;
			}
			else{
				rowlength=rowB-(cbnum-1)*BLOCK_SIZE;
			}
			for (int k=0;k<submatrixB[block].rowpointer[rowlength];k++)
			{
				BlockB_Val[vBid]=submatrixB[block].value[k];
				BlockB_Col[vBid]=submatrixB[block].columnindex[k];
				vBid++;
			}
			for (int jid=0;jid<rowlength;jid++)
			{
				BlockB_Ptr[pBid]=submatrixB[block].rowpointer[jid];
				pBid++;
			}
		}
	}
*/
/*		for (int i=0;i<nnzA;i++)
	{
		printf("%f     ",BlockB_Val[i]);
	}
	printf("\n");
	for (int i=0;i<nnzA;i++)
	{
		printf("%d     ",BlockB_Col[i]);
	}
	printf("\n");
	for (int i=0;i<ptrB_length;i++)
	{
		printf("%d     ",BlockB_Ptr[i]);
	}
	printf("\n");
*/
	

	for (int i=0;i<rbnum*cbnum;i++){
		if (nnzAnum[i]!=0){
			free(submatrixA[i].value);
			free(submatrixA[i].columnindex);
			free(submatrixA[i].rowpointer);
		}
	/*	if (nnzBnum[i]!=0){
			free(submatrixB[i].value);
			free(submatrixB[i].columnindex);
			free(submatrixB[i].rowpointer);
		}
	*/
	}

	
	// spmv using csr
	MAT_VAL_TYPE *x = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * colA);
	for (int i = 0; i < colA; i++)
	{
		x[i] = i % 10;
	}
	MAT_VAL_TYPE *y_golden = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * rowA);

	gettimeofday(&t1, NULL);
	for (int i = 0; i < BENCH_REPEAT; i++)
	{
		for (int i = 0; i < rowA; i++)
		{
			MAT_VAL_TYPE sum = 0;
			for (int j = matrixA.rowpointer[i]; j < matrixA.rowpointer[i+1]; j++)
			{
				sum += matrixA.value[j] * x[matrixA.columnindex[j]];
			}
			y_golden[i] = sum;
		}
	}
	gettimeofday(&t2, NULL);

    double time_csr_spmv = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	time_csr_spmv /= BENCH_REPEAT;
	printf("  CPU CSR SpMV %4.2f GFlops\n", 2 * (double)nnzA * 1.0e-6 / time_csr_spmv);

    // spmv using block csr
	MAT_VAL_TYPE *y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * rowA);

	// for each block row
	// parallel for
//	printf("rbnum=%d\n",rbnum);
/*	for(int i=0;i<rbnum+1;i++)
	{
		printf("%d    ",rowblock_ptr[i]);
	}
	printf("\n");
*/

    for (int blki = 0; blki < rbnum; blki++)
	{
		printf("rowblockid = %i, #blocks = %i\n", blki, rowblock_ptr[blki+1]-rowblock_ptr[blki]);
		for (int blkj = rowblock_ptr[blki]; blkj < rowblock_ptr[blki+1]; blkj++)
		{
			printf("%i, ", nnzb_A[blkj+1]-nnzb_A[blkj]);
		}
		printf("\n");
	}

	gettimeofday(&t1, NULL);
	for (int i = 0; i < BENCH_REPEAT; i++)
	{
		for (int blki = 0; blki < rbnum; blki++)
		{
			// clear y covered by the block row
		//	int blocksize;
		//	blocksize= blki == (rbnum-1) ? 
			for (int ri = 0; ri < BLOCK_SIZE; ri++)
			{
				y[blki * BLOCK_SIZE + ri] = 0;
			}

			// for each block in the block row
			for (int blkj = rowblock_ptr[blki]; blkj < rowblock_ptr[blki+1]; blkj++)
			{
				int x_offset = columnid[blkj] * BLOCK_SIZE;

				// for each row in the block
				for (int ri = 0; ri < BLOCK_SIZE; ri++)
				{
					MAT_VAL_TYPE sum = 0;
					// for each nonzero in the row of the block
					// the last row uses nnzlocal
					int stop = ri == BLOCK_SIZE - 1 ? (nnzb_A[blkj+1]-nnzb_A[blkj]) : BlockA_Ptr[ri+1+blkj*BLOCK_SIZE];
					for (int rj = BlockA_Ptr[blkj*BLOCK_SIZE+ri]; rj < stop; rj++)
					{
						sum += x[x_offset + BlockA_Col[nnzb_A[blkj]+rj]] * BlockA_Val[nnzb_A[blkj]+rj];
					}
					y[blki * BLOCK_SIZE + ri] += sum;
				}
			}
		}
	}
	gettimeofday(&t2, NULL);

    double time_ccsb_spmv = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	time_ccsb_spmv /= BENCH_REPEAT;
	printf("  CPU CCSB SpMV %4.2f GFlops\n", 2 * (double)nnzA * 1.0e-6 / time_ccsb_spmv);

    // check results
	int errcount = 0;
	for (int i = 0; i < rowA; i++)
	{
		if (y[i] != y_golden[i])
		{

			errcount++;
			//printf("%f    %f,%d\n",y[i],y_golden[i],i);
		}
	}
	printf("spmv errcount = %i\n", errcount);

    // run gpu 
    // set device
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    printf("---------------------------------------------------------------\n");
    printf("Device [ %i ] %s @ %4.2f MHz\n",
           device_id, deviceProp.name, deviceProp.clockRate * 1e-3f);

    ccsb_spmv_cuda(rowblock_ptr, columnid, nnzb_A, BlockA_Ptr, BlockA_Col, BlockA_Val,
    		   x, y_golden, rowA, colA, nnzA, rbnum, nnzbl, ptrA_length);

}
