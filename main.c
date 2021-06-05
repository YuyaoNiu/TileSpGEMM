#include"common.h"
#include"mmio_highlevel.h"
//#include"mmio.h"
#include"utils.h"
//#include<math.h>
//#include "hash.h"
//#include "thmkl_dcsrmultcsr_bin.h"
//#include <omp.h>

typedef struct 
{
	MAT_VAL_TYPE *value;
	int *columnindex;
	MAT_PTR_TYPE *rowpointer;
}SMatrix;

/*void writeresults(char *filename_res, char *filename, 
                  int mA, int nA, MAT_PTR_TYPE nnzA, 
                  double time, double gflops, int nthreads)
{
    printf("kernel = %s, nthreads = %i, time = %4.5f sec, gflops = %4.5f\n", 
           filename_res, nthreads, time, gflops);
    FILE *fres = fopen(filename_res, "a"); 
    if (fres == NULL) printf("Writing results fails.\n");
    fprintf(fres, "%s,%i,%i,%lld,%f,%f,%i\n", filename, mA, nA, nnzA, time, gflops, nthreads);
    fclose(fres);
}
*/
void writeresults(char *filename_res, char *filename, 
                  int mA, int nA, MAT_PTR_TYPE nnzA, 
                  int row,int col,MAT_PTR_TYPE subCub,double time)
{
  //  printf("kernel = %s, nthreads = %i, time = %4.5f sec, gflops = %4.5f\n", 
   //        filename_res, nthreads, time, gflops);
    FILE *fres = fopen(filename_res, "a"); 
    if (fres == NULL) printf("Writing results fails.\n");
    fprintf(fres, "%s,%i,%i,%lld,%i,%i,%i,%f\n", filename, mA, nA, nnzA, row,col,subCub,time);
    fclose(fres);
}

void DivideSub(SMatrix A,int start,int over,SMatrix submatrix[],int sub,int SubCol,int nnznum[])
{
	int i,j,k;
	j=A.rowpointer[start];
	int *num;
	num=(int*)malloc((SubNum)*sizeof(int));
	//count=(int*)malloc((SubNum)*sizeof(int));
	for (i=0;i<SubNum;i++){
		num[i]=0; 
	}
	
	int temp=1;
	for (i=start;i<over;i++)
	{
		while(j<A.rowpointer[i+1]){
			for (k=0;k<SubNum;k++){
				if (nnznum[sub+k]!=0){
					if (k==SubNum-1){
						num[k]++;
						submatrix[sub+k].value[num[k]-1]=A.value[j];
						submatrix[sub+k].columnindex[num[k]-1]=A.columnindex[j]-k*SubCol;
					}
					else if (A.columnindex[j]>=k*SubCol&&A.columnindex[j]<(k+1)*SubCol){
						num[k]++;
						submatrix[sub+k].value[num[k]-1]=A.value[j];
						submatrix[sub+k].columnindex[num[k]-1]=A.columnindex[j]-k*SubCol;
						break;
					}
				}
			}
			j++;
		}
		for (int p=0;p<SubNum;p++){
			if (nnznum[sub+p]!=0)
				submatrix[sub+p].rowpointer[temp]=num[p];
		}      
		temp++;
		//printf("%d\n",num[p]);
	}
	free(num);
}




void nnzNum(SMatrix A,int nnz[],int start,int over,int i,int SubCol)
{
	//printf("start=%d,over=%d\n",start,over);
	//printf("Subcol=%d\n",SubCol);

	int j;
	//int k=0;
	for (j=A.rowpointer[start];j<A.rowpointer[over];j++){
		for (int k=0;k<SubNum-1;k++){
			if (A.columnindex[j]>=k*SubCol&&A.columnindex[j]<(k+1)*SubCol){
				nnz[i+k]++;
				break;
			}
		}
	} 
	int m=0;
	for (int p=i;p<i+SubNum-1;p++)
		m+= nnz[p];
		

	nnz[i+SubNum-1]=A.rowpointer[over]-A.rowpointer[start]-m;
}


void block_mul( const int *flag, const int *mA, SMatrix *submatrixA,SMatrix *submatrixB,SMatrix *submatrixC,
                int blockCid,int *nnzAnum,int *nnzBnum,MAT_PTR_TYPE *subCub)
{
    int *num;
    num=(int*)malloc(((*mA)*SubNum)*sizeof(int));
	memset(num,0,((*mA)*SubNum)*sizeof(int));
	for (int colid=0;colid<SubNum;colid++)
	{
		int A=(blockCid/SubNum)*SubNum+colid;
		int B=blockCid%SubNum+colid*SubNum;
	//	printf("A=%d,nnz[A]=%d\n",A,nnzAnum[A]);
	//	printf("B=%d,nnz[B]=%d\n",B,nnzAnum[B]);
		int j=0;
		if (nnzAnum[A]!=0&&nnzBnum[B]!=0){
			for (int i=0;i<(*mA);i++){
				while(j<submatrixA[A].rowpointer[i+1])
				{	
					num[colid*(*mA)+i]+=submatrixB[B].rowpointer[submatrixA[A].columnindex[j]+1]-submatrixB[B].rowpointer[submatrixA[A].columnindex[j]];
					j++;
				}
			//	sum+=num[q*row+i];
			}
		}
	}
    int *rowCub; //calculate tasks for each row in subC
	rowCub=(int *)malloc((*mA)*sizeof(int));
	memset(rowCub,0,(*mA)*sizeof(int));
	for (int i=0;i<(*mA);i++)
	{
		for (int k=0;k<SubNum;k++)
			rowCub[i]+=num[k*(*mA)+i];
	}


	for(int i=0;i<(*mA);i++)
	{
		(*subCub)+=rowCub[i];
	}
/*	printf("The %d 's task\n",blockCid);
	for(int i=0;i<(*mA);i++)
	{
		printf("%d   ",rowCub[i]);
	}
	printf("\n");
*/	
	if (*flag==0)
	{
		for (int iid=0;iid<(*mA);iid++)
		{
			if (rowCub[iid]!=0)
			{

			int hashsize_full_reg=rowCub[iid];
			int *tmpIdx2D0 = (int *)malloc(hashsize_full_reg* sizeof(int));  //index 
			for (int l = 0; l < hashsize_full_reg; l++)  //hashsize_full_reg is the hash length assigned to the thread
				tmpIdx2D0[l] = -1;  //0x80000000=-1 ,pos is the starting position of the thread 
			for(int jid=0;jid<SubNum;jid++)
			{
				int A=(blockCid/SubNum)*SubNum+jid;  //SubmatrixA's id to be calculated
				int B=blockCid%SubNum+jid*SubNum;	 //SubmatrixB's id to be calculated
				if (nnzAnum[A]!=0&&nnzBnum[B]!=0){
					for(int i=submatrixA[A].rowpointer[iid];i<submatrixA[A].rowpointer[iid+1];i++)
					{
						int col=submatrixA[A].columnindex[i];
						for(int l=submatrixB[B].rowpointer[col];l<submatrixB[B].rowpointer[col+1];l++)
						{
							const int key = submatrixB[B].columnindex[l];
							int hashadr = (key*107) % hashsize_full_reg;
							while (1)
							{
								const int keyexist = tmpIdx2D0[hashadr]; //tmpIdx2Dthread[hashadr];
								if (keyexist == key)
								{
								//    tmpVal2D0[hashadr] +=b[l]*a[i] ;
									break;
								}
								else if (keyexist == -1)
								{
									tmpIdx2D0[hashadr] = key;
									submatrixC[blockCid].rowpointer[iid]++;
									break;
								}
								else
								{
									hashadr = (hashadr + 1) % hashsize_full_reg;
											// in step 1, it is not possible to overflow, since the assigned space is upper bound
								}
							}

						}
					}
				}
			}
			free(tmpIdx2D0);
		}
		}
		int new_val;
		int old_val;
		old_val=submatrixC[blockCid].rowpointer[0];
		submatrixC[blockCid].rowpointer[0]=0;
		for (int i = 1; i < (*mA)+1; i++)
		{   
			new_val = submatrixC[blockCid].rowpointer[i];
			submatrixC[blockCid].rowpointer[i] = old_val + submatrixC[blockCid].rowpointer[i - 1];
			old_val = new_val;
		}
	}
	else if (*flag==1)
	{
		for (int iid=0;iid<(*mA);iid++)
		{
			if (rowCub[iid]!=0){
			int hashsize_full_reg=rowCub[iid];
			int *tmpIdx2D0 = (int *)malloc(hashsize_full_reg* sizeof(int));  //index 
			MAT_VAL_TYPE *tmpVal2D0 = (MAT_VAL_TYPE *)malloc(hashsize_full_reg* sizeof(MAT_VAL_TYPE));  //value
            memset(tmpVal2D0,0,hashsize_full_reg* sizeof(MAT_VAL_TYPE));
			for (int l = 0; l < hashsize_full_reg; l++)  //hashsize_full_reg is the hash length assigned to the thread
				tmpIdx2D0[l] = -1;  //0x80000000=-1 ,pos is the starting position of the thread 
			for(int jid=0;jid<SubNum;jid++)
			{
				int A=(blockCid/SubNum)*SubNum+jid;  //SubmatrixA's id to be calculated
				int B=blockCid%SubNum+jid*SubNum;	 //SubmatrixB's id to be calculated
				if (nnzAnum[A]!=0&&nnzBnum[B]!=0){
					for(int i=submatrixA[A].rowpointer[iid];i<submatrixA[A].rowpointer[iid+1];i++)
					{
						int col=submatrixA[A].columnindex[i];
						for(int l=submatrixB[B].rowpointer[col];l<submatrixB[B].rowpointer[col+1];l++)
						{
							const int key = submatrixB[B].columnindex[l];
							int hashadr = (key*107) % hashsize_full_reg;
							while (1)
							{
								const int keyexist = tmpIdx2D0[hashadr]; //tmpIdx2Dthread[hashadr];
								if (keyexist == key)
								{
								    tmpVal2D0[hashadr] +=submatrixB[B].value[l]*submatrixA[A].value[i] ;
									break;
								}
								else if (keyexist == -1)
								{
									tmpIdx2D0[hashadr] = key;
									tmpVal2D0[hashadr] = submatrixB[B].value[l]*submatrixA[A].value[i];
								//	submatrixC[blockCid].rowpointer[iid]++;
									break;
								}
								else
								{
									hashadr = (hashadr + 1) % hashsize_full_reg;
											// in step 1, it is not possible to overflow, since the assigned space is upper bound
								}
							}

						}
					}
				}
			}
			int cptr=submatrixC[blockCid].rowpointer[iid];
            for (int k=0;k<hashsize_full_reg;k++)
            {
            	if (tmpIdx2D0[k]!=-1)
                {
                    submatrixC[blockCid].value[cptr]=tmpVal2D0[k];
                    submatrixC[blockCid].columnindex[cptr]=tmpIdx2D0[k];
                    cptr++;
                }
            }
			free(tmpIdx2D0);
			free(tmpVal2D0);
		}
		}
	}
	
}

int main(int argc, char ** argv)
{

	if (argc < 2)
    {
        printf("Run the code by './test matrix.mtx'.\n");
        return 0;
    }
	
    printf("--------------------------------!!!!!!!!------------------------------------\n");
//	int argi = 1;

 //   char  *filename1;
//	char  *filename2;
//    if(argc > argi)
//    {
//        filename1 = argv[argi];
 //       argi++;
 //   }

 	struct timeval t1, t2,t3,t4,t5,t6;
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

    if (rowA != colA)
    {
        printf("This code only computes square matrices.\n Exit.\n");
        return 0;
    }
	MAT_PTR_TYPE *cscColPtrA = (MAT_PTR_TYPE *)malloc((colA+1) * sizeof(MAT_PTR_TYPE));
    int *cscRowIdxA = (int *)malloc(nnzA   * sizeof(int));
    MAT_VAL_TYPE *cscValA    = (MAT_VAL_TYPE *)malloc(nnzA  * sizeof(MAT_VAL_TYPE));


	 // transpose A from csr to csc
    matrix_transposition(rowA, colA, nnzA, matrixA.rowpointer, matrixA.columnindex, matrixA.value,cscRowIdxA, cscColPtrA, cscValA);

	SMatrix matrixB;
	int rowB=colA;
	int colB=rowA;

	matrixB.rowpointer = cscColPtrA;
    matrixB.columnindex = cscRowIdxA;
    matrixB.value    = cscValA;

 /*   printf("--------------------------------!!!!!!!!------------------------------------\n");
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
//	printf("rowA=%d   ",rowA);
//	printf("colA=%d   ",colA);
//	printf("nnzA=%d\n",nnzA);

 	char  *filename;
    filename = argv[1];
    printf("MAT: -------------- %s --------------\n", filename);

	printf("input matrix A: ( %i, %i ) nnz = %i\n", rowA, colA, nnzA);

	
	SMatrix matrixA;
	matrixA.value=(MAT_VAL_TYPE*)malloc((nnzA)*sizeof(MAT_VAL_TYPE));
	matrixA.columnindex=(int*)malloc((nnzA)*sizeof(int));
	matrixA.rowpointer=(MAT_PTR_TYPE*)malloc((rowA+1)*sizeof(MAT_PTR_TYPE));

	
	mmio_data(matrixA.rowpointer,matrixA.columnindex,matrixA.value, filename1);
	
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

	printf("input matrix B: ( %i, %i ) nnz = %i\n", rowB, colB, nnzB);
	SMatrix matrixB;
	matrixB.value=(MAT_VAL_TYPE*)malloc((nnzB)*sizeof(MAT_VAL_TYPE));
	matrixB.columnindex=(int*)malloc((nnzB)*sizeof(int));
	matrixB.rowpointer=(MAT_PTR_TYPE*)malloc((rowB+1)*sizeof(MAT_PTR_TYPE));

	mmio_data(matrixB.rowpointer,matrixB.columnindex,matrixB.value, filename2);
	
/*	int threads_num=0;	

	if(argc > argi)
	{
		threads_num=atoi(argv[argi]);
		argi++;
	}
	printf("threads_num=%i\n",threads_num);

    // write results to text (scv) file
    FILE *fout = fopen("results.csv", "a");
    if (fout == NULL)
        printf("Writing results fails.\n");
    fprintf(fout, "%s,%s,%i,%i,%i,%i\n",
            filename1, filename2, rowA, rowB, nnzA, threads_num);
    fclose(fout);

	omp_set_num_threads(threads_num);

*/

/*	for (i=0;i<nnzB;i++)
	{
		printf("%f    ",matrixB.value[i]);
	}
	printf("\n");
	for (i=0;i<nnzB;i++)
	{
		printf("%d    ",matrixB.columnindex[i]);
	}
	printf("\n");

*/

	if (SubNum>rowA){
		printf("Error!\n");
		return 0;
	}
    
	int SubRowA=rowA/SubNum;
    int SubColA=colA/SubNum;
    int SubRowB=rowB/SubNum;
    int SubColB=colB/SubNum;
	
	SMatrix *submatrixA;
	SMatrix *submatrixB;
	SMatrix *submatrixC;
	submatrixA=(SMatrix*)malloc((SubNum*SubNum)*sizeof(SMatrix));
    
	submatrixB=(SMatrix*)malloc((SubNum*SubNum)*sizeof(SMatrix));

	submatrixC=(SMatrix*)malloc((SubNum*SubNum)*sizeof(SMatrix));

    int *nnzAnum,*nnzBnum;
	nnzAnum=(int*)malloc((SubNum*SubNum)*sizeof(int));
	nnzBnum=(int*)malloc((SubNum*SubNum)*sizeof(int));
	//nnzCnum=(int*)malloc((SubNum*SubNum)*sizeof(int));
   memset(nnzAnum,0,(SubNum*SubNum)*sizeof(int));
   memset(nnzBnum,0,(SubNum*SubNum)*sizeof(int));

	
//calculate nnz in each block

	#pragma omp parallel for

	for (int i=0;i<SubNum-1;i++)
	{
		nnzNum(matrixA,nnzAnum,i*SubRowA,(i+1)*SubRowA,i*SubNum,SubColA);
		nnzNum(matrixB,nnzBnum,i*SubRowB,(i+1)*SubRowB,i*SubNum,SubColB);
	}
	nnzNum(matrixA,nnzAnum,(SubNum-1)*SubRowA,rowA,(SubNum-1)*SubNum,SubColA);
	nnzNum(matrixB,nnzBnum,(SubNum-1)*SubRowB,rowB,(SubNum-1)*SubNum,SubColB);

/*	for (int i=0;i<SubNum*SubNum;i++)
	{
		writeresults("blocksize.csv", filename, rowA, colA,nnzA,i,nnzAnum[i]);
	}
*/

	for (int i=0;i<SubNum*SubNum;i++)
	{
		int rowlength;
		if (nnzAnum[i]!=0)
		{
			submatrixA[i].value=(MAT_VAL_TYPE*)malloc((nnzAnum[i])*sizeof(MAT_VAL_TYPE));
			submatrixA[i].columnindex=(int*)malloc((nnzAnum[i])*sizeof(int));
			if (i<SubNum*(SubNum-1)){
				rowlength=SubRowA;
			}
			else{
				rowlength=rowA-(SubNum-1)*SubRowA;
			}
			submatrixA[i].rowpointer=(MAT_PTR_TYPE *)malloc((rowlength+1)*sizeof(MAT_PTR_TYPE));
			memset(submatrixA[i].rowpointer,0,(rowlength+1)*sizeof(MAT_PTR_TYPE));
		}
		if (nnzBnum[i]!=0)
		{
			submatrixB[i].value=(MAT_VAL_TYPE*)malloc((nnzBnum[i])*sizeof(MAT_VAL_TYPE));
			submatrixB[i].columnindex=(int*)malloc((nnzBnum[i])*sizeof(int));
			if (i<SubNum*(SubNum-1)){
				submatrixB[i].rowpointer=(MAT_PTR_TYPE*)malloc((SubRowB+1)*sizeof(MAT_PTR_TYPE));
				memset(submatrixB[i].rowpointer,0,(SubRowB+1)*sizeof(MAT_PTR_TYPE));
			}
			else{
				submatrixB[i].rowpointer=(MAT_PTR_TYPE*)malloc((rowB-(SubNum-1)*SubRowB+1)*sizeof(MAT_PTR_TYPE));
				memset(submatrixB[i].rowpointer,0,(rowB-(SubNum-1)*SubRowB+1)*sizeof(MAT_PTR_TYPE));
			}
		}
	}

	for (int i=0;i<SubNum-1;i++)
	{
		DivideSub(matrixA,i*SubRowA,(i+1)*SubRowA,submatrixA,i*SubNum,SubColA,nnzAnum);
		DivideSub(matrixB,i*SubRowB,(i+1)*SubRowB,submatrixB,i*SubNum,SubColB,nnzBnum);
		
	}
	DivideSub(matrixA,(SubNum-1)*SubRowA,rowA,submatrixA,(SubNum-1)*SubNum,SubColA,nnzAnum);
	DivideSub(matrixB,(SubNum-1)*SubRowB,rowB,submatrixB,(SubNum-1)*SubNum,SubColB,nnzBnum);

	MAT_VAL_TYPE *BlockA_Val=(MAT_VAL_TYPE*)malloc((nnzA)*sizeof(MAT_VAL_TYPE));
	int  *BlockA_Col=(int*)malloc((nnzA)*sizeof(int));
	MAT_PTR_TYPE *BlockA_Ptr;
	int *BlockAid;
	int vAid=0;
	int pAid=0;
	int bnum=0,bAid=0;
	int ptrA_length=0;
	MAT_PTR_TYPE *bAnnzptr;
	for (int i=0;i<SubNum*SubNum;i++)
	{
		int rowlength;
		if (i<SubNum*(SubNum-1)){
			rowlength=SubRowA;
		}
		else{
			rowlength=rowA-(SubNum-1)*SubRowA;
		}
		if (nnzAnum[i]!=0){
			bnum++;
			ptrA_length+=(rowlength+1);
		}
	}
//	printf("bnum=%d\n",bnum);
	printf("ptr length=%d\n",ptrA_length);
	BlockA_Ptr=(MAT_PTR_TYPE*)malloc((ptrA_length)*sizeof(MAT_PTR_TYPE));
	BlockAid=(int*)malloc((bnum)*sizeof(int));
//	bAnnzptr=(MAT_PTR_TYPE*)malloc((bnum+1)*sizeof(MAT_PTR_TYPE));
//	memset(bAnnzptr,0,(bnum+1)*sizeof(MAT_PTR_TYPE));
	for (int i=0;i<SubNum*SubNum;i++)
	{
		int rowlength;
		if (i<SubNum*(SubNum-1)){
			rowlength=SubRowA;
		}
		else{
			rowlength=rowA-(SubNum-1)*SubRowA;
		}
		if (nnzAnum[i]!=0)
		{
			BlockAid[bAid]=i;
		//	bAnnzptr[bAid+1]=submatrixA[i].rowpointer[rowlength];
			bAid++;
			for (int j=0;j<submatrixA[i].rowpointer[rowlength];j++)
			{
				BlockA_Val[vAid]=submatrixA[i].value[j];
				BlockA_Col[vAid]=submatrixA[i].columnindex[j];
				vAid++;
			}
			for (int jid=0;jid<=rowlength;jid++)
			{
				BlockA_Ptr[pAid]=submatrixA[i].rowpointer[jid];
				pAid++;
			}
		}

	}

	MAT_VAL_TYPE *BlockB_Val=(MAT_VAL_TYPE*)malloc((nnzA)*sizeof(MAT_VAL_TYPE));
	int  *BlockB_Col=(int*)malloc((nnzA)*sizeof(int));
	MAT_PTR_TYPE *BlockB_Ptr;
	int *BlockBid;
	int vBid=0;
	int pBid=0;
	int ptrB_length=0;
	int bBid=0;
	for (int i=0;i<SubNum*SubNum;i++)
	{
		int rowlength;
		if (i<SubNum*(SubNum-1)){
			rowlength=SubRowB;
		}
		else{
			rowlength=rowB-(SubNum-1)*SubRowB;
		}
		if (nnzBnum[i]!=0){
			ptrB_length+=(rowlength+1);
		}
	}
	printf("ptrB length=%d\n",ptrB_length);
	BlockB_Ptr=(MAT_PTR_TYPE*)malloc((ptrB_length)*sizeof(MAT_PTR_TYPE));
	BlockBid=(int*)malloc((bnum)*sizeof(int));

	for (int i=0;i<SubNum;i++)
	{
		int rowlength;
		for (int j=0;j<SubNum;j++)
		{
			int id=i%SubNum+j*SubNum;
			if (id<SubNum*(SubNum-1)){
				rowlength=SubRowB;
			}
			else{
				rowlength=rowB-(SubNum-1)*SubRowB;
			}
			if (nnzBnum[id]!=0)
			{
				BlockBid[bBid]=id;
				bBid++;
				for (int jid=0;jid<submatrixB[id].rowpointer[rowlength];jid++)
				{
					BlockB_Val[vBid]=submatrixB[id].value[jid];
					BlockB_Col[vBid]=submatrixB[id].columnindex[jid];
					vBid++;
				}
				for (int jid=0;jid<=rowlength;jid++)
				{
					BlockB_Ptr[pBid]=submatrixB[id].rowpointer[jid];
					pBid++;
				}
			}
		}
	}

	for(int i=0;i<bnum;i++)
	{
		printf ("%d    ",BlockAid[i]);
	}


	unsigned long long int nnzCub = 0;
	for (int i = 0; i < nnzA; i++)
    {
        int rowB = matrixA.columnindex[i];
        nnzCub += matrixB.rowpointer[rowB + 1] - matrixB.rowpointer[rowB];
    }
    double flops = 2.0 * nnzCub / 1.0e9; // flop mul-add for each nonzero entry
    printf("SpGEMM flops = %lld.\n", nnzCub);

	
	for (int nthreads = 1; nthreads <= NTHREADS_MAX; nthreads *= 2)
    {
	//	for (int ri = 0; ri < BENCH_REPEAT; ri++)
    //    {

			double gflops = 0;
			double time=0;
			MAT_PTR_TYPE nnzC=0;
			
			omp_set_num_threads(nthreads);
			printf("\n#threads = %i\n", nthreads);

			gettimeofday(&t1, NULL);

			#pragma omp parallel for
			for (int i=0;i<SubNum*SubNum;i++)
			{
			//	gettimeofday(&t1, NULL);
				int flag;
				int rowlength,collength;
				flag=0;
				MAT_PTR_TYPE subCub=0;
				if (i<SubNum*(SubNum-1)){
					rowlength=SubRowA;
					collength=SubColA;
				}	
				else{
					rowlength=rowA-(SubNum-1)*SubRowA;
					collength=colA-(SubNum-1)*SubColA;
				}
				submatrixC[i].rowpointer=(MAT_PTR_TYPE*)malloc((rowlength+1)*sizeof(MAT_PTR_TYPE));
				memset(submatrixC[i].rowpointer,0,(rowlength+1)*sizeof(MAT_PTR_TYPE));
			//	printf("submatrixC %d\n",i);
			
				block_mul(&flag,&rowlength,submatrixA,submatrixB, submatrixC,i,nnzAnum,nnzBnum,&subCub);
			//	printf("%d\n",subCub);
				if (subCub!=0)
				{
					int subnnz=submatrixC[i].rowpointer[rowlength];
					submatrixC[i].value=(MAT_VAL_TYPE *)malloc(subnnz*sizeof(MAT_VAL_TYPE));
					submatrixC[i].columnindex=(int *)malloc(subnnz*sizeof(int));
					flag=1;
					subCub=0;
					block_mul(&flag,&rowlength,submatrixA,submatrixB, submatrixC,i,nnzAnum,nnzBnum,&subCub);
				}

				if (i==329&&nthreads==1)
				{
				//	printf("???????????\n");
					freopen("webbasse_block 10*9.mtx","w",stdout);
    				printf("%d %d %d\n", rowlength, collength, submatrixC[i].rowpointer[rowlength]);

   					for (int iid = 1; iid <= rowlength; iid++)
    				{
        				for (int j = submatrixC[i].rowpointer[iid-1]; j < submatrixC[i].rowpointer[iid]; j++)
        				{
							printf("%d %d %f\n", iid, submatrixC[i].columnindex[j],submatrixC[i].value[j]);
						}
					}
					fclose(stdout);
				}
				
			//	gettimeofday(&t2, NULL);
			//	time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
			//	time = time / 1000.0; // a single runtime, in seconds
				
			//	if (nthreads==1)

			//	writeresults("blocksize-offshore-32*32.mtx.csv", filename, rowA, colA, nnzA,(i/SubNum),(i%SubNum),subCub,time);



			/*	for (int iid=0;iid<rowlength+1;iid++)
				{
					printf("%d   ",submatrixC[i].rowpointer[iid]);
				}
				printf("\n");
			*/
				nnzC+=submatrixC[i].rowpointer[rowlength];
			}
			
			for (int i=0;i<SubNum*SubNum;i++)
			{
				free(submatrixC[i].value);
				free(submatrixC[i].columnindex);
				free(submatrixC[i].rowpointer);
			}
			
			printf("nnzC=%d\n",nnzC);
	//	}
		gettimeofday(&t2, NULL);
		time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		time = time / 1000.0; // a single runtime, in seconds
		printf("time = %4.5f sec\n",time);
		gflops = flops / time; // Gflop/s
		printf("gflops=%4.5f\n",gflops);
	//	writeresults("block_mul.csv", filename, rowA, colA, nnzA, time, gflops, nthreads);
	}

	for (int i=0;i<SubNum*SubNum;i++){
		if (nnzAnum[i]!=0){
			free(submatrixA[i].value);
			free(submatrixA[i].columnindex);
			free(submatrixA[i].rowpointer);
		}
		if (nnzBnum[i]!=0){
			free(submatrixB[i].value);
			free(submatrixB[i].columnindex);
			free(submatrixB[i].rowpointer);
		}
	}
	free(submatrixA);
	free(submatrixB);
	free(submatrixC);
	free(nnzAnum);
	free(nnzBnum);

	free(matrixA.value);
	free(matrixA.columnindex);
	free(matrixA.rowpointer);
	free(matrixB.value);
	free(matrixB.columnindex);
	free(matrixB.rowpointer);


/*	for (int i=0;i<SubNum*SubNum;i++)
	{
		printf("submatrixC %d\n",i);
		if (i<SubNum*(SubNum-1))
		{
			for (int j=0;j<submatrixC[i].rowpointer[SubRowA];j++)
			{
				printf("%f    ",submatrixC[i].value[j]);
			}
			printf("\n");
			for (int j=0;j<submatrixC[i].rowpointer[SubRowA];j++)
			{
				printf("%d    ",submatrixC[i].columnindex[j]);
			}
			printf("\n");

		}
		else
		{
			for (int j=0;j<submatrixC[i].rowpointer[rowA-(SubNum-1)*SubRowA];j++)
			{
				printf("%f    ",submatrixC[i].value[j]);
			}
			printf("\n");
		}
		
	}
*/
	

	

}
