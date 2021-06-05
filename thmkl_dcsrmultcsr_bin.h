//#include"common.h"
#include <stdio.h>
#include <stdlib.h>

void thmkl_dcsrmultcsr(const char *trans, const int *request, const int *sort, 
                       const int *m, const int *n, const int *k, float *a, int*ja, 
                       int *ia, float *b, int *jb, int *ib, float *c, int *jc,
                       int *ic, const int *nzmax, int *info )
{   
    int nthreads = omp_get_max_threads();

    int *rowCub = (int *)malloc((*m) * sizeof(int));
    memset(rowCub,0,(*m) * sizeof(int));
    for (int i=0;i<*m;i++)
    {
        for(int j=ia[i];j<ia[i+1];j++)
        {
            int rowB=ja[j];
            rowCub[i]+=ib[rowB+1]-ib[rowB];
        }
    }
   

    int *bin;
    bin=(int *)malloc((9*(*m))*sizeof(int));
 //   int bin[9*(*m)]; 

    int *flag=(int*)malloc(9*sizeof(int)); 
 //   int flag[9];
    memset(flag,0,9*sizeof(int));


    int *method=(int*)malloc(9*sizeof(int));
 //   int method[9];
    memset(method,0,9*sizeof(int));
    method[8]=2;
 /*   for (int i=0;i<10;i++)
    {
        method[i]=1;
    }
*/
    for(int i=0;i<*m;i++)
    {
        if (rowCub[i]==0){
            bin[0*(*m)+flag[0]]=i;flag[0]++;
        }
        else if (rowCub[i]==1){
            bin[1*(*m)+flag[1]]=i;flag[1]++;
        }
         else if (rowCub[i]>=2&&rowCub[i]<=32){
            bin[2*(*m)+flag[2]]=i;flag[2]++;
        }  
         else if (rowCub[i]>=33&&rowCub[i]<=64){
            bin[3*(*m)+flag[3]]=i;flag[3]++;
        }
         else if (rowCub[i]>=65&&rowCub[i]<=128){
            bin[4*(*m)+flag[4]]=i;flag[4]++;
        }
         else if (rowCub[i]>=129&&rowCub[i]<=256){
            bin[5*(*m)+flag[5]]=i;flag[5]++;
        }
         else if (rowCub[i]>=257&&rowCub[i]<=512){
            bin[6*(*m)+flag[6]]=i;flag[6]++;
        }
      //  if(rowCub[i]>=513&&rowCub[i]<=1024){
       //     bin[7*(*m)+flag[7]]=i;flag[7]++;
       // }
      //  if(rowCub[i]>=1025&&rowCub[i]<=2048){
       //     bin[8*(*m)+flag[8]]=i;flag[8]++;
       // }
         else if(rowCub[i]>512&&rowCub[i]<=30000){
            bin[7*(*m)+flag[7]]=i;flag[7]++;
        }
        else if(rowCub[i]>30000){
            bin[8*(*m)+flag[8]]=i;flag[8]++;
        }

           
    }
    
 /*   if (nthreads==1&&*request==1)
    {
        for (int i=0;i<8;i++)
    {
        printf("bin[%d]:",i);
        for (int j=0;j<flag[i];j++)
        {
            printf("%d   ",bin[i*(*m)+j]);
        }
        printf("\n");

    }
*/
   // if (nthreads==1&&*request==1)

 /*   for (int i=0;i<10;i++)
    {
        printf("flag[%d]=%d\n",i,flag[i]);
    }
*/
 
    
  
if(*request==1)
{
    struct timeval t1, t2;
  //  printf(">>>>>>>>>>>>>>>>>>>>>>>>>request=1>>>>>>>>>>>>>>>>>>>>>>>>\n");
    for(int bid=0;bid<9;bid++)
    {
        if (flag[bid]==0)
            continue;
        gettimeofday(&t1, NULL);
        int start=bid*(*m);
       // printf("bin[%d],start=%d\n",bid,start);
        int end=bid*(*m)+flag[bid];
       // printf("bin[%d],end=%d\n",bid,end);
        if (bid==0)
        {
            #pragma omp parallel for
           // int start=bid*(*m);
           // int end=bid*(*m)+flag[bid]-1;
            for (int i=start;i<end;i++)
            {
                ic[bin[i]]=0;
            }
        }
        else if(bid==1)
        {
            #pragma omp parallel for
            for (int i=start;i<end;i++)
            {
                ic[bin[i]]=1;
            }
        }
        else 
        {
            //int f=flag[bid];
            nthreads = nthreads >flag[bid] ? flag[bid]: nthreads;
            int threadrow=flag[bid]/nthreads;
            int hashsize_full_reg;
            if(bid>1&&bid<7)
            {
                hashsize_full_reg=8;
                for (int i=0;i<bid;i++)
                    hashsize_full_reg*=2;
            }
            else if (bid==7||bid==8)
            {
                hashsize_full_reg=0;
             
                for (int j=start;j<end;j++)
                {    
                    if (rowCub[bin[j]]>hashsize_full_reg)
                        hashsize_full_reg=rowCub[bin[j]];
                }
            }
            hashsize_full_reg = hashsize_full_reg > *k ? *k : hashsize_full_reg;
         //   printf("hashsize_full_reg=%d\n",hashsize_full_reg);
           
           #pragma omp parallel for
            for (int tid=0;tid<nthreads;tid++)
            {
                int rowstart=bid*(*m)+tid*threadrow;
                int rowend;
                if (tid<nthreads-1)
                {   
                    rowend=bid*(*m)+(tid+1)*threadrow;
                }
                else if(tid==nthreads-1)
                {
                    rowend=bid*(*m)+flag[bid];
                }
            //    if (bid<=6)
            //        int tmpIdx2D0[hashsize_full_reg];
            //    else if (bid==7||bid==8)
                    int *tmpIdx2D0 = (int *)malloc(hashsize_full_reg* sizeof(int));  //index 
                if (method[bid]==0)
                {
                    for (int l = 0; l < hashsize_full_reg; l++)  //hashsize_full_reg is the hash length assigned to the thread
                    tmpIdx2D0[l] = -1;  //0x80000000=-1 ,pos is the starting position of the thread 

                    for (int iid=rowstart;iid<rowend;iid++)
                    {
                        int j=bin[iid]; 
                        for(int i=ia[j];i<ia[j+1];i++)
                        {
                            int col=ja[i];
                            for(int l=ib[col];l<ib[col+1];l++)
                            {
                                const int key = jb[l];
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
                                        ic[j]++;
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
                    //    printf("ic[%d]=%d\n",j,ic[j]);
                        for (int l = 0; l < hashsize_full_reg; l++)  //hashsize_full_reg is the hash length assigned to the thread
                            tmpIdx2D0[l] = -1;  //0x80000000=-1 ,pos is the starting position of the thread 
                    
                    }
                }
                else if (method[bid]==1)
                {
                    for (int iid=rowstart;iid<rowend;iid++)
                    {
                        int pos=0;
                        int j=bin[iid]; 
                        for(int i=ia[j];i<ia[j+1];i++)
                        {
                            int col=ja[i];
                            for(int l=ib[col];l<ib[col+1];l++)
                            {
                                const int key = jb[l];
                                tmpIdx2D0[pos]=key;
                                pos++;
                                
                            }
                        }
                        quick_sort_key(tmpIdx2D0, pos);
                        ic[j]=1;
                        for (int pi=1;pi<pos;pi++)
                        {
                            if(tmpIdx2D0[pi]!=tmpIdx2D0[pi-1])
                            {
                                ic[j]++;
                            }
                        }
                    }
                }
                else if (method[bid]==2)
                {
                //    if (bid<=6)
                //    {
                //        char d_dense_row_column_flag[*k];
                //    }
                //    else {
                        char *d_dense_row_column_flag = (char *)malloc((*k)*sizeof(char));
                //    }
                    memset(d_dense_row_column_flag,0,(*k)*sizeof(char));
                    
                    for (int iid=rowstart;iid<rowend;iid++)
                    {
                       // int pos=0;
                        int j=bin[iid]; 
                        for(int i=ia[j];i<ia[j+1];i++)
                        {
                            int col=ja[i];
                            for(int l=ib[col];l<ib[col+1];l++)
                            {
                                const int key = jb[l];
                                //tmpIdx2D0[pos]=key;
                                d_dense_row_column_flag[key]=1;
                            //    pos++;
                                
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
                        ic[j] = nnzr;
                        memset(d_dense_row_column_flag,0,(*k)*sizeof(char));
                    }
                //    if (bid==7||bid==8)
                        free(d_dense_row_column_flag);

                }
            //    if (bid==7||bid==8)
                    free(tmpIdx2D0);
            }
        }
     //   gettimeofday(&t2, NULL);
     //   double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
     //   printf("bin %d used  %4.5f sec\n",bid,time/1000.0);
           
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
}

if(*request==2)
{
     struct timeval t1, t2;
    
   // printf(">>>>>>>>>>>>>>>>>>>>>>>>>request=2>>>>>>>>>>>>>>>>>>>>>>>>\n");
    for(int bid=0;bid<9;bid++)
    {
        if (flag[bid]==0)
            continue;
        gettimeofday(&t1, NULL);
        int start=bid*(*m);
        int end=bid*(*m)+flag[bid];
     /*  if (bid==0)
        {
            #pragma omp parallel for
            for (int i=bin[bid*(*m)+0];i<bin[bid*(*m)+flag[bid]];i++)
            {
                ic[i]=0;
            }
        }
    */
        if(bid==1)
        {
            #pragma omp parallel for
            for (int i=start;i<end;i++)
            {
                int cid=bin[i];
                c[ic[cid]]=a[ia[cid]]*b[ib[ja[ia[cid]]]];
                jc[ic[cid]]=jb[ib[ja[ia[cid]]]];
            }
        }
        else
        {
            nthreads = nthreads > flag[bid] ? flag[bid] : nthreads;
            int threadrow=flag[bid]/nthreads;
            int hashsize_full_reg;
            if(bid>1&&bid<7)
            {
                hashsize_full_reg=8;
                for (int i=0;i<bid;i++)
                    hashsize_full_reg*=2;
               
            }
            else if (bid==7||bid==8)
            {
                hashsize_full_reg=0;
             
                for (int j=start;j<end;j++)
                {    
                    if (rowCub[bin[j]]>hashsize_full_reg)
                        hashsize_full_reg=rowCub[bin[j]];
                }

            }
             hashsize_full_reg = hashsize_full_reg > *k ? *k : hashsize_full_reg;
           
         //   printf("hashsize_full_reg=%d\n",hashsize_full_reg);
            #pragma omp parallel for
            for (int tid=0;tid<nthreads;tid++)
            {
                int rowstart=bid*(*m)+tid*threadrow;
                int rowend;
                if (tid<nthreads-1)
                {   
                    rowend=bid*(*m)+(tid+1)*threadrow;
                }
                else if(tid==nthreads-1)
                {
                    rowend=bid*(*m)+flag[bid];
                }

            //    MAT_VAL_TYPE tmpVal2D0[hashsize_full_reg];
             //   int tmpIdx2D0[hashsize_full_reg];
                VALUE_TYPE *tmpVal2D0 = (VALUE_TYPE *)malloc(hashsize_full_reg* sizeof(VALUE_TYPE));  //value
                memset(tmpVal2D0,0,hashsize_full_reg* sizeof(VALUE_TYPE));
                int *tmpIdx2D0 = (int *)malloc(hashsize_full_reg* sizeof(int));  //index 

                if (method[bid]==0)
                {
                    for (int l = 0; l < hashsize_full_reg; l++)  //hashsize_full_reg is the hash length assigned to the thread
                    tmpIdx2D0[l] = -1;  //0x80000000=-1 ,pos is the starting position of the thread 

                    for (int iid=rowstart;iid<rowend;iid++)
                    {
                        int j=bin[iid];
                        for(int i=ia[j];i<ia[j+1];i++)
                        {
                            int col=ja[i];
                            for(int l=ib[col];l<ib[col+1];l++)
                            {
                                const int key = jb[l];
                                int hashadr = (key*107) % hashsize_full_reg;
                                while (1)
                                {
                                    const int keyexist = tmpIdx2D0[hashadr]; //tmpIdx2Dthread[hashadr];
                                    if (keyexist == key)
                                    {
                                        tmpVal2D0[hashadr] +=b[l]*a[i] ;
                                        break;
                                    }
                                    else if (keyexist == -1)
                                    {
                                        tmpIdx2D0[hashadr] = key;
                                        tmpVal2D0[hashadr] = b[l]*a[i];
                                        //hashsize_real_local[j]++;
                                    //   ic[j]++;
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
                    int cptr=ic[j];
                    for (int k=0;k<hashsize_full_reg;k++)
                    {
                        if (tmpIdx2D0[k]!=-1)
                        {
                            c[cptr]=tmpVal2D0[k];
                            jc[cptr]=tmpIdx2D0[k];
                            cptr++;
                        }
                    }
                    memset(tmpVal2D0,0,hashsize_full_reg* sizeof(VALUE_TYPE));
                    for (int l = 0; l < hashsize_full_reg; l++)  //hashsize_full_reg is the hash length assigned to the thread
                        tmpIdx2D0[l] = -1;  //0x80000000=-1 ,pos is the starting position of the thread 
                    }

                }
                else if (method[bid]==1)
                {
                     for (int iid=rowstart;iid<rowend;iid++)
                    {
                        int pos=0;
                        int j=bin[iid]; 
                        for(int i=ia[j];i<ia[j+1];i++)
                        {
                            int col=ja[i];
                            for(int l=ib[col];l<ib[col+1];l++)
                            {
                                const int key = jb[l];
                                tmpIdx2D0[pos]=key;
                                tmpVal2D0[pos]=b[l]*a[i];
                                pos++;
                                
                            }
                        }
                     //   quick_sort_key(tmpIdx2D0, pos);
                        quick_sort_key_val_pair(tmpIdx2D0, tmpVal2D0, pos);
                     //   ic[j]=1;
                     int ci=ic[j];
                     c[ci]=tmpVal2D0[0];
                     jc[ci]=tmpIdx2D0[0];
                        for (int pi=1;pi<pos;pi++)
                        {
                            if(tmpIdx2D0[pi]!=tmpIdx2D0[pi-1])
                            {
                                ci++;
                                c[ci]=tmpVal2D0[pi];
                                jc[ci]=tmpIdx2D0[pi];
                               
                            }
                            else{
                                c[ci]+=tmpVal2D0[pi];
                            }
                        }
                    }
                    
                }
                else if (method[bid]==2)
                {
                 //   char d_dense_row_column_flag[*k];
                 //   MAT_VAL_TYPE d_dense_row_value[*k];
                    char *d_dense_row_column_flag = (char *)malloc((*k)*sizeof(char));
                    VALUE_TYPE *d_dense_row_value = (VALUE_TYPE *)malloc((*k)*sizeof(VALUE_TYPE));
                    memset(d_dense_row_column_flag, 0, (*k)*sizeof(char));
                    memset(d_dense_row_value, 0, (*k)*sizeof(VALUE_TYPE));
                    for (int iid=rowstart;iid<rowend;iid++)
                    {
                        int pos=0;
                        int j=bin[iid]; 
                        for(int i=ia[j];i<ia[j+1];i++)
                        {
                            int col=ja[i];
                            for(int l=ib[col];l<ib[col+1];l++)
                            {
                                const int key = jb[l];
                                //tmpIdx2D0[pos]=key;
                                d_dense_row_column_flag[key]=1;
                                d_dense_row_value[key]+=b[l]*a[i];
                                pos++;
                            }
                        }
                   
                    int nnzr = ic[j];
                    for (int cid = 0; cid < *k; cid++)
                    {   
                        if (d_dense_row_column_flag[cid] == 1)
                        {
                            c[nnzr]= d_dense_row_value[cid];
                            jc[nnzr]=cid;
                            nnzr++;
                        }
                    }
                    ic[j] = nnzr;
                    memset(d_dense_row_column_flag,0,(*k)*sizeof(char));
                    memset(d_dense_row_value,0,(*k)*sizeof(VALUE_TYPE));
                    }
                    free(d_dense_row_column_flag);
                    free(d_dense_row_value);
                }
                
                
                free(tmpIdx2D0);
                free(tmpVal2D0);
            }
        }
     //   gettimeofday(&t2, NULL);
     //   double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
     //   printf("bin %d used  %4.5f sec\n",bid,time/1000.0);
    }
}  
free(rowCub);
free(flag);
free(bin);
free(method);
       
}