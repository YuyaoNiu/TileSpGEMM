#include"common.h"
#include"mmio_highlevel.h"
//#include"mmio.h"
#include "utils.h"
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

void writeresults(char *filename_res, char *filename, 
                  int mA, int nA, MAT_PTR_TYPE nnzA, 
                  double time, double gflops)
{
  //  printf("kernel = %s, nthreads = %i, time = %4.5f sec, gflops = %4.5f\n", 
  //         filename_res, nthreads, time, gflops);
    FILE *fres = fopen(filename_res, "a"); 
    if (fres == NULL) printf("Writing results fails.\n");
    fprintf(fres, "%s,%i,%i,%d,%f,%f\n", filename, mA, nA, nnzA, time, gflops);
    fclose(fres);
}


void writeresults1(char *filename_res, char *filename, 
                  int mA, int nA, MAT_PTR_TYPE nnzA, 
                  int Aid,int Bid,int rowlength,int collength,double cpratio,double time, double gflops)
{
  //  printf("kernel = %s, nthreads = %i, time = %4.5f sec, gflops = %4.5f\n", 
   //        filename_res, nthreads, time, gflops);
    FILE *fres = fopen(filename_res, "a"); 
    if (fres == NULL) printf("Writing results fails.\n");
    fprintf(fres, "%s,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f\n", filename, mA, nA, nnzA,Aid,Bid,rowlength ,collength,cpratio,time, gflops);
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


void block_mul( const int *flag, const int *mA, SMatrix *submatrixA,SMatrix *submatrixB,SMatrix *Ci,
                int blockCid,int blockci_id,int *nnzAnum,int *nnzBnum,unsigned long long int *cbub)
{
	*cbub=0;
    int *Ciub;
    Ciub=(int*)malloc((*mA)*sizeof(int));
	memset(Ciub,0,(*mA)*sizeof(int));

	int A=(blockCid/SubNum)*SubNum+blockci_id;
	int B=blockCid%SubNum+blockci_id*SubNum;
	//	printf("A=%d,nnz[A]=%d\n",A,nnzAnum[A]);
	//	printf("B=%d,nnz[B]=%d\n",B,nnzAnum[B]);
	int j=0;
	if (nnzAnum[A]!=0&&nnzBnum[B]!=0){
		for (int i=0;i<(*mA);i++){
			while(j<submatrixA[A].rowpointer[i+1])
			{	
				Ciub[i]+=submatrixB[B].rowpointer[submatrixA[A].columnindex[j]+1]-submatrixB[B].rowpointer[submatrixA[A].columnindex[j]];
				j++;
			}
			//	sum+=num[q*row+i];
		}
	}
	for(int i=0;i<(*mA);i++)
	{
		*cbub+=Ciub[i];

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
			int hashsize_full_reg=Ciub[iid];
			int *tmpIdx2D0 = (int *)malloc(hashsize_full_reg* sizeof(int));  //index 
			for (int l = 0; l < hashsize_full_reg; l++)  //hashsize_full_reg is the hash length assigned to the thread
				tmpIdx2D0[l] = -1;  //0x80000000=-1 ,pos is the starting position of the thread 
			
			int A=(blockCid/SubNum)*SubNum+blockci_id;  //SubmatrixA's id to be calculated
			int B=blockCid%SubNum+blockci_id*SubNum;	 //SubmatrixB's id to be calculated
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
								Ci[blockci_id].rowpointer[iid]++;
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
			free(tmpIdx2D0);
		}
		int new_val;
		int old_val;
		old_val=Ci[blockci_id].rowpointer[0];
		Ci[blockci_id].rowpointer[0]=0;
		for (int i = 1; i < (*mA)+1; i++)
		{   
			new_val = Ci[blockci_id].rowpointer[i];
			Ci[blockci_id].rowpointer[i] = old_val + Ci[blockci_id].rowpointer[i - 1];
			old_val = new_val;
		}
	}
	else if (*flag==1)
	{
		for (int iid=0;iid<(*mA);iid++)
		{
			int hashsize_full_reg=Ciub[iid];
			int *tmpIdx2D0 = (int *)malloc(hashsize_full_reg* sizeof(int));  //index 
			MAT_VAL_TYPE *tmpVal2D0 = (MAT_VAL_TYPE *)malloc(hashsize_full_reg* sizeof(MAT_VAL_TYPE));  //value
            memset(tmpVal2D0,0,hashsize_full_reg* sizeof(MAT_VAL_TYPE));
			for (int l = 0; l < hashsize_full_reg; l++)  //hashsize_full_reg is the hash length assigned to the thread
				tmpIdx2D0[l] = -1;  //0x80000000=-1 ,pos is the starting position of the thread 
			
			int A=(blockCid/SubNum)*SubNum+blockci_id;  //SubmatrixA's id to be calculated
			int B=blockCid%SubNum+blockci_id*SubNum;	 //SubmatrixB's id to be calculated
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
			int cptr=Ci[blockci_id].rowpointer[iid];
            for (int k=0;k<hashsize_full_reg;k++)
            {
            	if (tmpIdx2D0[k]!=-1)
                {
                    Ci[blockci_id].value[cptr]=tmpVal2D0[k];
                    Ci[blockci_id].columnindex[cptr]=tmpIdx2D0[k];
                    cptr++;
                }
            }
			free(tmpIdx2D0);
			free(tmpVal2D0);
		}

	}

}

void block_add(const int *flag, const int *mA, SMatrix *Ci,int blockCid,SMatrix *submatrixC)
{
	int *Cub;
    Cub=(int*)malloc((*mA)*sizeof(int));
	memset(Cub,0,(*mA)*sizeof(int));

	for (int jid=0;jid<(*mA);jid++)
	{
		for (int iid=0;iid<SubNum;iid++)
		{
			Cub[jid]+=Ci[iid].rowpointer[jid+1]-Ci[iid].rowpointer[jid];
		}

	}
	if (*flag==0)
	{
		for (int iid=0;iid<(*mA);iid++)
		{
		//	printf("Cub %d=%d\n",iid,Cub[iid]);
			int hashsize_full_reg=Cub[iid];
			int *tmpIdx2D0 = (int *)malloc(hashsize_full_reg* sizeof(int));  //index 
			for (int l = 0; l < hashsize_full_reg; l++)  //hashsize_full_reg is the hash length assigned to the thread
				tmpIdx2D0[l] = -1;  //0x80000000=-1 ,pos is the starting position of the thread 
			for(int jid=0;jid<SubNum;jid++)
			{
				for(int i=Ci[jid].rowpointer[iid];i<Ci[jid].rowpointer[iid+1];i++)
				{
					const int key = Ci[jid].columnindex[i];
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
			free(tmpIdx2D0);
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
		//	printf("Cub %d=%d\n",iid,Cub[iid]);
			int hashsize_full_reg=Cub[iid];
			int *tmpIdx2D0 = (int *)malloc(hashsize_full_reg* sizeof(int));  //index 
			MAT_VAL_TYPE *tmpVal2D0 = (MAT_VAL_TYPE *)malloc(hashsize_full_reg* sizeof(MAT_VAL_TYPE));  //value
            memset(tmpVal2D0,0,hashsize_full_reg* sizeof(MAT_VAL_TYPE));
			for (int l = 0; l < hashsize_full_reg; l++)  //hashsize_full_reg is the hash length assigned to the thread
				tmpIdx2D0[l] = -1;  //0x80000000=-1 ,pos is the starting position of the thread 
			for(int jid=0;jid<SubNum;jid++)
			{
				for(int i=Ci[jid].rowpointer[iid];i<Ci[jid].rowpointer[iid+1];i++)
				{
					const int key = Ci[jid].columnindex[i];
					int hashadr = (key*107) % hashsize_full_reg;
					while (1)
					{
						const int keyexist = tmpIdx2D0[hashadr]; //tmpIdx2Dthread[hashadr];
						if (keyexist == key)
						{
							tmpVal2D0[hashadr] +=Ci[jid].value[i] ;
						//    tmpVal2D0[hashadr] +=b[l]*a[i] ;
							break;
						}
						else if (keyexist == -1)
						{
							tmpIdx2D0[hashadr] = key;
							tmpVal2D0[hashadr] = Ci[jid].value[i];
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


/*	mmio_info(&rowA, &colA, &nnzA, &isSymmetricA, filename1);
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
	MAT_PTR_TYPE nnzB;
	int isSymmetricB;
	mmio_info(&rowB, &colB, &nnzB, &isSymmetricB, filename2);

	printf("input matrix B: ( %i, %i ) nnz = %i\n", rowB, colB, nnzB);
	SMatrix matrixB;
	matrixB.value=(MAT_VAL_TYPE*)malloc((nnzB)*sizeof(MAT_VAL_TYPE));
	matrixB.columnindex=(int*)malloc((nnzB)*sizeof(int));
	matrixB.rowpointer=(MAT_PTR_TYPE*)malloc((rowB+1)*sizeof(MAT_PTR_TYPE));

	mmio_data(matrixB.rowpointer,matrixB.columnindex,matrixB.value, filename2);

*/
	
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

	int *nnzCnum;
	nnzCnum=(int *)malloc((SubNum*SubNum)*sizeof(int));

	//struct timeval t1, t2;
	
	
	
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
		MAT_PTR_TYPE nnzC=0;
		double gflops = 0;
		double time=0;
		double time_s=0;
		double time_n=0;
		omp_set_num_threads(nthreads);
        printf("\n#threads = %i\n", nthreads);

		gettimeofday(&t5, NULL);

		#pragma omp parallel for
		for (int i=0;i<SubNum*SubNum;i++)
		{
			int flag;
			int rowlength;
			int collength;
			unsigned long long int cbub=0;
			double cbflops;
			
			int subCinnz=0;
			int subCnnz=0;
			if (i<SubNum*(SubNum-1)){
				rowlength=SubRowA;
				collength=SubColA;

			}	
			else{
				rowlength=rowA-(SubNum-1)*SubRowA;
				collength=colA-(SubNum-1)*SubColA;
			}
			SMatrix *Ci=(SMatrix*)malloc((SubNum)*sizeof(SMatrix));
			submatrixC[i].rowpointer=(MAT_PTR_TYPE*)malloc((rowlength+1)*sizeof(MAT_PTR_TYPE));
			memset(submatrixC[i].rowpointer,0,(rowlength+1)*sizeof(MAT_PTR_TYPE));
		//	printf("submatrixC %d\n",i);
			for (int j=0;j<SubNum;j++)
			{
				double cpratio=0;
				gettimeofday(&t1, NULL);
				cbub=0;
				flag=0;
				Ci[j].rowpointer=(MAT_PTR_TYPE*)malloc((rowlength+1)*sizeof(MAT_PTR_TYPE));
				memset(Ci[j].rowpointer,0,(rowlength+1)*sizeof(MAT_PTR_TYPE));
				block_mul(&flag,&rowlength,submatrixA,submatrixB, Ci,i,j,nnzAnum,nnzBnum,&cbub);
				gettimeofday(&t2, NULL);
				time_s = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
				time_s = time_s / 1000.0; // a single runtime, in seconds
				cbflops = 2.0 * cbub / 1.0e9; // flop mul-add for each nonzero entry
				subCinnz=Ci[j].rowpointer[rowlength];
				gflops = cbflops / time_s;
				if(subCinnz!=0)
				{
					cpratio=cbub/subCinnz;
					if (nthreads==1)
						writeresults1("block-symbolic32*32.csv", filename, rowA, colA, nnzA, (i/SubNum)*SubNum+j,i%SubNum+j*SubNum,rowlength,collength,cpratio,time_s,gflops);
					
				}
				
				
					
				//	printf("subCinnz %d=%d\n",j,subCinnz);
				gettimeofday(&t3, NULL);
				Ci[j].value=(MAT_VAL_TYPE *)malloc(subCinnz*sizeof(MAT_VAL_TYPE));
				Ci[j].columnindex=(int *)malloc(subCinnz*sizeof(int));
				flag=1;
				block_mul(&flag,&rowlength,submatrixA,submatrixB, Ci,i,j,nnzAnum,nnzBnum,&cbub);

				gettimeofday(&t4, NULL);
				time_n = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
				time_n = time_n / 1000.0; // a single runtime, in seconds
			//	printf("time = %4.5f sec\n",time_n);
				gflops = cbflops / time_n; // Gflop/s
				//	printf("gflops=%4.5f\n",gflops);
				if(subCinnz!=0)
				{
					cpratio=cbub/subCinnz;
					if (nthreads==1)
						writeresults1("block-numerical32*32.csv", filename, rowA, colA, nnzA, (i/SubNum)*SubNum+j,i%SubNum+j*SubNum,rowlength,collength,cpratio,time_n,gflops);
				}
			
				
			}
		//	printf("????????????\n");
			flag=0;
			block_add(&flag,&rowlength,Ci,i,submatrixC);
			subCnnz=submatrixC[i].rowpointer[rowlength];
		//	printf("subCnnz %d=%d\n",i,subCnnz);
			submatrixC[i].value=(MAT_VAL_TYPE *)malloc(subCnnz*sizeof(MAT_VAL_TYPE));
			submatrixC[i].columnindex=(int *)malloc(subCnnz*sizeof(int));
			flag=1;
			block_add(&flag,&rowlength,Ci,i,submatrixC);

			for (int j=0;j<SubNum;j++)
			{
				free(Ci[j].value);
				free(Ci[j].rowpointer);
				free(Ci[j].columnindex);
			}
			free(Ci);
		//	nnzC+=subCnnz;
		}

		for(int i=0;i<SubNum*SubNum;i++)
		{
			int rowlength=0;
			if (i<SubNum*(SubNum-1)){
				rowlength=SubRowA;
			}	
			else{
				rowlength=rowA-(SubNum-1)*SubRowA;
			}
			nnzC+=submatrixC[i].rowpointer[rowlength];

		}
		printf("nnzC=%d\n",nnzC);
		gettimeofday(&t6, NULL);
		time = (t6.tv_sec - t5.tv_sec) * 1000.0 + (t6.tv_usec - t5.tv_usec) / 1000.0;
		time = time / 1000.0; // a single runtime, in seconds
	//	printf("time = %4.5f sec\n",time);
		gflops = flops / time; // Gflop/s
	//	printf("gflops=%4.5f\n",gflops);
		writeresults("block_mul32*32.csv", filename, rowA, colA, nnzA, time, gflops);

//	}

/*	MAT_PTR_TYPE nnzC=0;
	for(int i=0;i<SubNum*SubNum;i++)
	{
		int r;
		if (i<SubNum*(SubNum-1))
			r=SubRowA;
		else
		{
			r=rowA-(SubNum-1)*SubRowA;
		}
		nnzC+=submatrixC[i].rowpointer[r];
		
	}
	printf("nnzC=%d\n",nnzC);
*/
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
}
