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
	//rowA=128;
	//nnzA=matrixA.rowpointer[128];
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

   
    int *flag=(int *)malloc(cbnum*sizeof(int));
    int nnzbl=0;

    for (int i=0;i<rbnum;i++)
	{
        memset(flag,0,cbnum*sizeof(int));
        int start= i*BLOCK_SIZE;
        int end = i==rbnum-1 ?  rowA : (i+1)*BLOCK_SIZE ;
		for (int j=matrixA.rowpointer[start];j<matrixA.rowpointer[end];j++){
            int jc=matrixA.columnindex[j]/BLOCK_SIZE;
            if (flag[jc]==0)
            {
                flag[jc]=1;
                nnzbl++;
            }
	    } 
	}
 //   printf("nnzbl=%d\n",nnzbl);

    MAT_PTR_TYPE *rowblock_ptr;    //block rowpointer of A
	int *columnid;                // block columnindex of A
	int *nnzb_A;
	int colid=0;
	rowblock_ptr=(MAT_PTR_TYPE *)malloc((rbnum+1)*sizeof(MAT_PTR_TYPE));
    columnid=(int *)malloc(nnzbl*sizeof(int));
	
    memset(rowblock_ptr,0,(rbnum+1)*sizeof(MAT_PTR_TYPE));
    int ptrA_length=0;
    for (int i=0;i<rbnum;i++)
	{
        memset(flag,0,cbnum*sizeof(int));
        int start= i*BLOCK_SIZE;
        int end = i==rbnum-1 ?  rowA : (i+1)*BLOCK_SIZE ;
		for (int j=matrixA.rowpointer[start];j<matrixA.rowpointer[end];j++){
            int jc=matrixA.columnindex[j]/BLOCK_SIZE;
            if (flag[jc]==0)
            {
                flag[jc]=1;
                rowblock_ptr[i+1]++;
                columnid[colid]=jc;
                colid++;
                ptrA_length+=(end-start);
            }
	    } 
	}
    for (int i=1;i<rbnum+1;i++)
	{
		rowblock_ptr[i]+=rowblock_ptr[i-1];
	}

//    exclusive_scan(nnzb_A,nnzbl+1);

/*    for (int i=0;i<rbnum+1;i++)
	{
		printf("%d    ",rowblock_ptr[i]);
	}
    printf("\n");

	for (int i=0;i<nnzbl;i++)
	{
		printf("%d    ",columnid[i]);
	}
    printf("\n");
*/
    MAT_VAL_TYPE *BlockA_Val=(MAT_VAL_TYPE*)malloc((nnzA)*sizeof(MAT_VAL_TYPE));
	unsigned char  *BlockA_Col=(unsigned char*)malloc((nnzA)*sizeof(unsigned char));
	unsigned char *BlockA_Ptr=(unsigned char*)malloc((ptrA_length)*sizeof(unsigned char));

    nnzb_A=(int *)malloc((nnzbl+1)*sizeof(int));
	int vAid=0;
	int pAid=0;
    int nnzid=0;

//    int *rownnzA=(int *)malloc(cbnum*sizeof(int));
//    SMatrix *subrowmatrixA=(SMatrix *)malloc(cbnum*sizeof(SMatrix));

    //for each row block
    for (int i=0;i<rbnum;i++)
	{
        int rowbnum=rowblock_ptr[i+1]-rowblock_ptr[i];
     //   printf("rowbnum=%d\n",rowbnum);
        int *rownnzA=(int *)malloc(rowbnum*sizeof(int));
        SMatrix *subrowmatrixA=(SMatrix *)malloc(rowbnum*sizeof(SMatrix));
        memset(rownnzA,0,rowbnum*sizeof(int));

        int rowlength= i==rbnum-1 ? rowA-(rbnum-1)*BLOCK_SIZE : BLOCK_SIZE ;
     //   printf("rowlength=%d\n",rowlength);
        int start= i*BLOCK_SIZE;
        int end = i==rbnum-1 ?  rowA : (i+1)*BLOCK_SIZE ;
		for (int j=matrixA.rowpointer[start];j<matrixA.rowpointer[end];j++)
        {
            int ki;
            for (int k=rowblock_ptr[i],ki=0;k<rowblock_ptr[i+1],ki<rowbnum;k++,ki++)
            {
                int kcstart=columnid[k]*BLOCK_SIZE;
                int kcend= columnid[k]== (cbnum-1) ?  colA: (columnid[k]+1)*BLOCK_SIZE;
                if (matrixA.columnindex[j]>=kcstart&&matrixA.columnindex[j]<kcend)
                {
                    rownnzA[ki]++;
                    break;
                }
		    }
	    } 
     /*   for (int p=0;p<rowbnum;p++)
        {
            printf("%d    ",rownnzA[p]);
        }
        printf("\n");
    */
        for (int bi=0;bi<rowbnum;bi++) 
        {
           subrowmatrixA[bi].value=(MAT_VAL_TYPE*)malloc((rownnzA[bi])*sizeof(MAT_VAL_TYPE));
           subrowmatrixA[bi].columnindex=(int *)malloc((rownnzA[bi])*sizeof(int));
          
           subrowmatrixA[bi].rowpointer=(MAT_PTR_TYPE *)malloc((rowlength+1)*sizeof(MAT_PTR_TYPE));
           memset(subrowmatrixA[bi].rowpointer,0,(rowlength+1)*sizeof(MAT_PTR_TYPE));
        }
        int *num=(int*)malloc((rowbnum)*sizeof(int));
	//count=(int*)malloc((SubNum)*sizeof(int));
	    memset(num,0,(rowbnum)*sizeof(int));
    //    int temp=1;
    
     //   int blocksize= i==rbnum-1 ? rowA-i*BLOCK_SIZE : BLOCK_SIZE ;
        for (int ri=0;ri<rowlength;ri++)
        {
            for (int j=matrixA.rowpointer[start+ri];j<matrixA.rowpointer[start+ri+1];j++)
            {
                int ki;
                for (int k=rowblock_ptr[i],ki=0;k<rowblock_ptr[i+1],ki<rowbnum;k++,ki++)
                {
                    int kcstart=columnid[k]*BLOCK_SIZE;
                    int kcend= columnid[k]== (cbnum-1) ?  colA: (columnid[k]+1)*BLOCK_SIZE;
                    if (matrixA.columnindex[j]>=kcstart&&matrixA.columnindex[j]<kcend)
                    {
                        num[ki]++;
                        subrowmatrixA[ki].value[num[ki]-1]=matrixA.value[j];
                        subrowmatrixA[ki].columnindex[num[ki]-1]=matrixA.columnindex[j]-columnid[k]*BLOCK_SIZE;
                        break;
                    }
                }
            }
            for (int bi=0;bi<rowbnum;bi++){
                    subrowmatrixA[bi].rowpointer[ri+1]=num[bi];
            }   
	    }

    /*    for(int iid=0;iid<rowbnum;iid++)
        {
            for (int jid=0;jid<rownnzA[iid];jid++)
            {
                printf("%f     ",subrowmatrixA[iid].value[jid]);
            }
            printf("\n");
        }
    */
        for(int bi=0;bi<rowbnum;bi++)
        {
            nnzb_A[nnzid]=rownnzA[bi];
            nnzid++;
        }
    
        for(int bi=0;bi<rowbnum;bi++)
        {
            for (int k=0;k<rownnzA[bi];k++)
            {
                BlockA_Val[vAid]=subrowmatrixA[bi].value[k];
                BlockA_Col[vAid]=subrowmatrixA[bi].columnindex[k];
                vAid++;
            }
            for (int jid=0;jid<rowlength;jid++)
            {
                BlockA_Ptr[pAid]=subrowmatrixA[bi].rowpointer[jid];
                pAid++;
            }
        }

        free(rownnzA);
        for (int bi=0;bi<rowbnum;bi++)
        {
            free(subrowmatrixA[bi].value);
            free(subrowmatrixA[bi].columnindex);
            free(subrowmatrixA[bi].rowpointer);
        }
        free(subrowmatrixA);
        free(num);
    }

    exclusive_scan(nnzb_A,nnzbl+1);
    
 /*   printf("BlockA_Val\n");
	for (int i=0;i<nnzA;i++)
	{
		printf("%f     ",BlockA_Val[i]);
	}
	printf("\n");
	printf("\n");

	printf("BlockA_Col\n");
	for (int i=0;i<nnzA;i++)
	{
		printf("%d     ",BlockA_Col[i]);
	}
	printf("\n");
	printf("\n");
	printf("BlockA_Ptr\n");
	for (int i=0;i<ptrA_length;i++)
	{
		printf("%d     ",BlockA_Ptr[i]);
	}
	printf("\n");
*/

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
printf("-=-=-=-=-=-=-=-=-\n");
    // run gpu 
    // set device
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
printf("-=-=-=-=-=-=-=-=-\n");
    printf("---------------------------------------------------------------\n");
    printf("Device [ %i ] %s @ %4.2f MHz\n",
           device_id, deviceProp.name, deviceProp.clockRate * 1e-3f);
printf("-=-=-=-=-=-=-=-=-\n");
    ccsb_spmv_cuda(rowblock_ptr, columnid, nnzb_A, BlockA_Ptr, BlockA_Col, BlockA_Val,
    		   x, y_golden, rowA, colA, nnzA, rbnum, nnzbl, ptrA_length);



}
