#include"common.h"
#include"mmio_highlevel.h"
#include"mmio.h"
#include <omp.h>
#define SubNum 2
typedef struct 
{
	double *value;
	int *columnindex;
	int *rowpointer;
}SMatrix;

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
void ESC(int rowptr,SMatrix Crowt,int over,int nnzcrow[])
{
	int i,j,k,tc;
	double tv;
	int num1=1,temp1=0;
	int count;

 	for (i=0;i<over-1;i++)
	{
		for (j=0;j<over-i-1;j++)
		{
			if (Crowt.columnindex[j]>Crowt.columnindex[j+1])
			{
				tc=Crowt.columnindex[j];tv=Crowt.value[j];
				Crowt.columnindex[j]=Crowt.columnindex[j+1];Crowt.value[j]=Crowt.value[j+1];
				Crowt.columnindex[j+1]=tc;Crowt.value[j+1]=tv;
			}
		}
	}
/*	for (i=0;i<over;i++)
	{
		printf("%f ",Crowt.value[i]);
	}
		printf("\n");
		for (i=0;i<over;i++)
	{
		printf("%d ",Crowt.columnindex[i]);
	}
	printf("\n");
*/
//	printf("over=%d\n",over);
//printf("%d  \n",nnzcrow[rowptr]);
	i=0;
	while(temp1<=over-1)
	{
		for (count=temp1;count<over-1;count++){
			if(Crowt.columnindex[temp1]==Crowt.columnindex[num1])
			{
				num1++;nnzcrow[rowptr]--;
			}
			else break;
		}
		Crowt.columnindex[i]=Crowt.columnindex[temp1];
		double sumtemp=0;
		for (k=temp1;k<num1;k++)
		{
			sumtemp+=Crowt.value[k];
		}
		Crowt.value[i]=sumtemp;
		temp1=num1;
		num1++;
		i++;
	}
	//printf("%d\n",num[rowptr+1]);
/*	for (i=0;i<nnzcrow[rowptr];i++)
	{
		printf("%f ",Crowt.value[i]);
	}
		printf("\n");
		for (i=0;i<nnzcrow[rowptr];i++)
	{
		printf("%d ",Crowt.columnindex[i]);
	}
	printf("\n");

	printf("%dth row Sort end!\n",rowptr);
*/
} 


int  MM_mul(SMatrix submatrixCt[],int p,SMatrix submatrixA[],SMatrix submatrixB[],int row,int nnzAnum[],int nnzBnum[])
{
	SMatrix Crowt;
	SMatrix Crow;
	int i,j,temp,k;
	int q;
	int sum=0;
	int count1=0;
	int *num;
	int A,B;
	int flag;

	num=(int *)malloc((row*SubNum)*sizeof(int));
	for (i=0;i<row*SubNum;i++)
		num[i]=0;
	for (q=0;q<SubNum;q++)
	{
		A=(p/SubNum)*SubNum+q;
		B=p%SubNum+q*SubNum;
		//printf("A=%d,nnz[A]=%d\n",A,nnzAnum[A]);
		//printf("B=%d,nnz[B]=%d\n",B,nnzAnum[B]);
		j=0;
		if (nnzAnum[A]!=0&&nnzBnum[B]!=0){
			for (i=0;i<row;i++){
				while(j<submatrixA[A].rowpointer[i+1])
				{	
					num[q*row+i]+=submatrixB[B].rowpointer[submatrixA[A].columnindex[j]+1]-submatrixB[B].rowpointer[submatrixA[A].columnindex[j]];
					j++;
				}
				sum+=num[q*row+i];
			}
		}
	}
	//if (sum!=0)
	//printf("sum %d =%d\n",p,sum);
   	if (sum!=0){
		flag=1;
		submatrixCt[p].value=(double*)malloc((sum)*sizeof(double));
		submatrixCt[p].columnindex=(int*)malloc((sum)*sizeof(int));
		submatrixCt[p].rowpointer=(int*)malloc((row+1)*sizeof(int));
		for (i=0;i<sum;i++){
			submatrixCt[p].value[i]=0;
			submatrixCt[p].columnindex[i]=0;
		}
		for (i=0;i<row+1;i++)
			submatrixCt[p].rowpointer[i]=0;
	
		int *nnzcrow;
		nnzcrow=(int *)malloc((row)*sizeof(int));
		for (i=0;i<row;i++)
			nnzcrow[i]=0;
		for (i=0;i<row;i++)
		{
			for (k=0;k<SubNum;k++)
				nnzcrow[i]+=num[k*row+i];
		}
	/*	for (i=0;i<row;i++){
			printf("the %d Crow  %d\n",i,nnzcrow[i]);
		}
	*/
		for (i=0;i<row;i++)
		{
			Crowt.value=(double*)malloc((nnzcrow[i])*sizeof(double));
			Crowt.columnindex=(int*)malloc((nnzcrow[i])*sizeof(int));
		//	Crow.value=(double*)malloc((nnzcrow[i])*sizeof(double));
		//	Crow.columnindex=(int*)malloc((nnzcrow[i])*sizeof(int));
			for (int m=0;m<nnzcrow[i];m++)
			{
				Crowt.value[m]=0;Crowt.columnindex[m]=0;
			//	Crow.value[m]=0;Crow.columnindex[m]=0;
			}
			k=0;
			for (q=0;q<SubNum;q++)
			{	
				A=(p/SubNum)*SubNum+q;
				B=p%SubNum+q*SubNum;
				if (nnzAnum[A]!=0&&nnzBnum[B]!=0){
					j=submatrixA[(p/SubNum)*SubNum+q].rowpointer[i];
					while(j<submatrixA[(p/SubNum)*SubNum+q].rowpointer[i+1])
					{	
						for (temp=submatrixB[B].rowpointer[submatrixA[A].columnindex[j]];temp<submatrixB[B].rowpointer[submatrixA[A].columnindex[j]+1];temp++,k++)
						{
							Crowt.value[k]=submatrixA[A].value[j]*submatrixB[B].value[temp];
							Crowt.columnindex[k]=submatrixB[B].columnindex[temp];
						}
						j++;
					}
				}
			}
	/*	for (int s=0;s<nnzcrow[i];s++){
			printf("%f  ",Crowt.value[s]);
		}
		printf("\n");
		for (int s=0;s<nnzcrow[i];s++){
			printf("%d  ",Crowt.columnindex[s]);
		}
		printf("\n");
	*/
			ESC(i,Crowt,nnzcrow[i],nnzcrow);
			
		//	free(Crowt.value);
		//	free(Crowt.columnindex);

			for (int m=0;m<nnzcrow[i];m++)
			{
				if (Crowt.value[m]!=0) 
				{
					submatrixCt[p].value[count1]=Crowt.value[m];
					submatrixCt[p].columnindex[count1]=Crowt.columnindex[m];
					count1++;
				}
			}
			submatrixCt[p].rowpointer[i+1]=count1;
			free(Crowt.value);
			free(Crowt.columnindex);
			
			
		}
/*		printf("the %d\n",p);
	for (i=0;i<count1;i++)
	{
		printf ("%f ",submatrixC[p].value[i]);
	}
	printf("\n");
	for (i=0;i<count1;i++)
	{
		printf ("%d ",submatrixC[p].columnindex[i]);
	}
	printf("\n");
	for (i=0;i<row+1;i++)
	{
		printf ("%d ",submatrixC[p].rowpointer[i]);
	}
	printf("\n");
*/		free(num);
		return flag;
	}
	else{
		flag=0;
		free(num);
		return flag;
	}
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


int main(int argc, char ** argv)
{
    printf("--------------------------------!!!!!!!!------------------------------------\n");
	int argi = 1;
	int i,j;

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
	
	SMatrix matrixA;
	matrixA.value=(double*)malloc((nnzA)*sizeof(double));
	matrixA.columnindex=(int*)malloc((nnzA)*sizeof(int));
	matrixA.rowpointer=(int*)malloc((rowA+1)*sizeof(int));

	
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


	printf("rowB=%d   ",rowB);
	printf("colB=%d   ",colB);
	printf("nnzB=%d\n",nnzB);

	SMatrix matrixB;
	matrixB.value=(double*)malloc((nnzB)*sizeof(double));
	matrixB.columnindex=(int*)malloc((nnzB)*sizeof(int));
	matrixB.rowpointer=(int*)malloc((rowB+1)*sizeof(int));

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
	SMatrix *submatrixCt;
	submatrixA=(SMatrix*)malloc((SubNum*SubNum)*sizeof(SMatrix));
    
	submatrixB=(SMatrix*)malloc((SubNum*SubNum)*sizeof(SMatrix));

	submatrixCt=(SMatrix*)malloc((SubNum*SubNum)*sizeof(SMatrix));

    int *nnzAnum,*nnzBnum;
	nnzAnum=(int*)malloc((SubNum*SubNum)*sizeof(int));
	nnzBnum=(int*)malloc((SubNum*SubNum)*sizeof(int));
	//nnzCnum=(int*)malloc((SubNum*SubNum)*sizeof(int));
    for (i=0;i<SubNum*SubNum;i++){
		nnzAnum[i]=0;nnzBnum[i]=0;
	}
	
//calculate nnz in each block
	for (i=0;i<SubNum-1;i++)
	{
		nnzNum(matrixA,nnzAnum,i*SubRowA,(i+1)*SubRowA,i*SubNum,SubColA);
		nnzNum(matrixB,nnzBnum,i*SubRowB,(i+1)*SubRowB,i*SubNum,SubColB);
	}
	nnzNum(matrixA,nnzAnum,i*SubRowA,rowA,i*SubNum,SubColA);
	nnzNum(matrixB,nnzBnum,i*SubRowB,rowB,i*SubNum,SubColB);

/*	int *blockflag;
	blockflag=(int *)malloc((SubNum*SubNum)*sizeof(int));
	for (i=0;i<SubNum*SubNum;i++)
	{
		if (nnzAnum[i]!=0)
			blockflag[i]=1;
		else
			blockflag[i]=1;
		
	}
*/
/*	for (i=0;i<SubNum*SubNum;i++){
		printf("%d ",nnzAnum[i]);
	}
	printf("\n");
*/	
/*	for (i=0;i<SubNum-1;i++)
	{
		nnzNum(matrixB,nnzBnum,i*SubRowB,(i+1)*SubRowB,i*SubNum,SubColB);
	}
	nnzNum(matrixB,nnzBnum,i*SubRowB,rowB,i*SubNum,SubColB);
*/
/*	for (i=0;i<SubNum*SubNum;i++){
		printf("%d ",nnzBnum[i]);
	}
*/

    for (i=0;i<SubNum*SubNum;i++)
	{
		if (nnzAnum[i]!=0)
		{
			submatrixA[i].value=(double*)malloc((nnzAnum[i])*sizeof(double));
			submatrixA[i].columnindex=(int*)malloc((nnzAnum[i])*sizeof(int));

		//	for (j=0;j<nnzAnum[i];j++){
		//		submatrixA[i].value[j]=0;
        //    	submatrixA[i].columnindex[j]=0;
		//	}
			if (i<SubNum*(SubNum-1)){
				submatrixA[i].rowpointer=(int*)malloc((SubRowA+1)*sizeof(int));
				for (j=0;j<SubRowA+1;j++)
				{
					submatrixA[i].rowpointer[j]=0;
				}
			}
			else{
				submatrixA[i].rowpointer=(int*)malloc((rowA-(SubNum-1)*SubRowA+1)*sizeof(int));
				for (j=0;j<rowA-(SubNum-1)*SubRowA+1;j++)
				{
					submatrixA[i].rowpointer[j]=0;
				}
			}

		}
		if (nnzBnum[i]!=0)
		{
			submatrixB[i].value=(double*)malloc((nnzBnum[i])*sizeof(double));
        	submatrixB[i].columnindex=(int*)malloc((nnzBnum[i])*sizeof(int));
		//	for (j=0;j<nnzBnum[i];j++){
        //    	submatrixB[i].value[j]=0;
        //    	submatrixB[i].columnindex[j]=0;
        //	}
			if (i<SubNum*(SubNum-1)){
				submatrixB[i].rowpointer=(int*)malloc((SubRowB+1)*sizeof(int));
				for (j=0;j<SubRowB+1;j++)
				{
					submatrixB[i].rowpointer[j]=0;
				}
			}
			else{
				submatrixB[i].rowpointer=(int*)malloc((rowB-(SubNum-1)*SubRowB+1)*sizeof(int));
	    		for (j=0;j<rowB-(SubNum-1)*SubRowB+1;j++)
				{
					submatrixB[i].rowpointer[j]=0;
				}
			}


		}
	}

	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

//Divide Matrix into n*n blocks
for (i=0;i<SubNum-1;i++)
	{
		DivideSub(matrixA,i*SubRowA,(i+1)*SubRowA,submatrixA,i*SubNum,SubColA,nnzAnum);
		DivideSub(matrixB,i*SubRowB,(i+1)*SubRowB,submatrixB,i*SubNum,SubColB,nnzBnum);
		
	}
	DivideSub(matrixA,i*SubRowA,rowA,submatrixA,i*SubNum,SubColA,nnzAnum);
	DivideSub(matrixB,i*SubRowB,rowB,submatrixB,i*SubNum,SubColB,nnzBnum);

/*	for (i=0;i<SubNum*SubNum;i++)
    {
		//printf("%d\n",nnzAnum[i]);
        printf("the %d :",i);

        for (j=0;j<nnzAnum[i];j++)
        {
            printf("%f ",submatrixA[i].value[j]);

        }
        printf("\n");

    }
	for (i=0;i<SubNum*SubNum;i++)
    {
        printf("the %d :",i);

        for (j=0;j<nnzAnum[i];j++)
        {
            printf("%d ",submatrixA[i].columnindex[j]);

        }
        printf("\n");

    }
	for (i=0;i<SubNum*SubNum;i++)
    {
		if (nnzAnum[i]!=0){
			printf("the %d :",i);
		if (i<SubNum*(SubNum-1))
        for (j=0;j<SubRowA+1;j++)
        {
            printf("%d ",submatrixA[i].rowpointer[j]);
        }
		else 
		for (j=0;j<rowA-(SubNum-1)*SubRowA+1;j++)
        {
            printf("%d ",submatrixA[i].rowpointer[j]);
        }
		
        printf("\n");
			
		}
        
	}

*/
int *subCtflag;
subCtflag=(int *)malloc((SubNum*SubNum)*sizeof(int));

#pragma omp parallel for
	for (i=0;i<SubNum*SubNum;i++)
	{
		if (i<SubNum*(SubNum-1)){
			subCtflag[i]=MM_mul(submatrixCt,i,submatrixA,submatrixB,SubRowA,nnzAnum,nnzBnum);
		}
		else
			subCtflag[i]=MM_mul(submatrixCt,i,submatrixA,submatrixB,rowA-(SubNum-1)*SubRowA,nnzAnum,nnzBnum);
	}
//	printf("!!!!!!!!!!!!!!!!!!!!!!\n");


	int subCnum=0;

	for (i=0;i<SubNum*SubNum;i++)
	{
		if (i<SubNum*(SubNum-1)&&subCtflag[i]==1){
			if (submatrixCt[i].rowpointer[SubRowA]!=0)
				subCnum++;
		}
		else if (i>=SubNum*(SubNum-1)&&subCtflag[i]==1){
			if (submatrixCt[i].rowpointer[rowA-(SubNum-1)*SubRowA]!=0)
				subCnum++;
		}
			
	}
	printf("SubCnum=%d\n",subCnum);
	SMatrix *submatrixC;
	submatrixC=(SMatrix*)malloc((subCnum)*sizeof(SMatrix));

	j=0;
	int nnzC=0;
	for (i=0;i<SubNum*SubNum;i++)
	{
		if (i<SubNum*(SubNum-1)&&subCtflag[i]==1){
			if (submatrixCt[i].rowpointer[SubRowA]!=0){
				submatrixC[j].value=(double*)malloc((submatrixCt[i].rowpointer[SubRowA])*sizeof(double));
				submatrixC[j].columnindex=(int*)malloc((submatrixCt[i].rowpointer[SubRowA])*sizeof(int));
				submatrixC[j].rowpointer=(int*)malloc((SubRowA+1)*sizeof(int));
				for (int k=0;k<submatrixCt[i].rowpointer[SubRowA];k++){
					submatrixC[j].value[k]=submatrixCt[i].value[k];
					submatrixC[j].columnindex[k]=submatrixCt[i].columnindex[k];
				}
				for (int k=0;k<SubRowA+1;k++)
					submatrixC[j].rowpointer[k]=submatrixCt[i].rowpointer[k];
				j++;
				nnzC+=submatrixC[j].rowpointer[SubRowA];
			}
		}
		else if (i>=SubNum*(SubNum-1)&&subCtflag[i]==1){
			if (submatrixCt[i].rowpointer[rowA-(SubNum-1)*SubRowA]!=0){
				submatrixC[j].value=(double*)malloc((submatrixCt[i].rowpointer[rowA-(SubNum-1)*SubRowA])*sizeof(double));
				submatrixC[j].columnindex=(int*)malloc((submatrixCt[i].rowpointer[rowA-(SubNum-1)*SubRowA])*sizeof(int));
				submatrixC[j].rowpointer=(int*)malloc((rowA-(SubNum-1)*SubRowA+1)*sizeof(int));
				for (int k=0;k<submatrixCt[i].rowpointer[rowA-(SubNum-1)*SubRowA];k++){
					submatrixC[j].value[k]=submatrixCt[i].value[k];
					submatrixC[j].columnindex[k]=submatrixCt[i].columnindex[k];
				}
				for (int k=0;k<rowA-(SubNum-1)*SubRowA+1;k++)
					submatrixC[j].rowpointer[k]=submatrixCt[i].rowpointer[k];
				j++;
				nnzC+=submatrixC[j].rowpointer[rowA-(SubNum-1)*SubRowA];
			}
		}

	}


	

	gettimeofday(&t2, NULL);
	double time_parallel = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("block spgemm used %4.2f ms\n", time_parallel);

/*	FILE *fout = fopen("results.csv", "a");
    if (fout == NULL)
        printf("Writing results fails.\n");
    fprintf(fout, "%s,%s,%i,%i,%i,%d,%4.2f ms\n",
            filename1, filename2, rowA, rowB, nnzA, SubNum*SubNum,time_serial);
    fclose(fout);
*/

/*	int nnzC=0;
	for (i=0;i<SubNum*SubNum;i++){
		if (subCflag[i]==1){
			if (i<SubNum*(SubNum-1)){
				nnzC+=submatrixC[i].rowpointer[SubRowA];
			}
			else{
				nnzC+=submatrixC[i].rowpointer[rowA-SubRowA*(SubNum-1)];
			}
		}
	}
		
	printf("nnzC=%d\n",nnzC);
*/
printf("nnzC=%d\n",nnzC);
		


	for (i=0;i<SubNum*SubNum;i++){
		if (subCtflag[i]==1)
		{
			free(submatrixCt[i].value);
			free(submatrixCt[i].columnindex);
			free(submatrixCt[i].rowpointer);
		}
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

	free(subCtflag);
	for (i=0;i<subCnum;i++){
		free(submatrixC[i].value);
		free(submatrixC[i].columnindex);
		free(submatrixC[i].rowpointer);
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





	
}
