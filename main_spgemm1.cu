#include"common.h"
#include"mmio_highlevel.h"
//#include"mmio.h"
#include"utils.h"
//#include "encode.h"
#include"utils_cuda_scan.h"

#include "spgemm_nsparse_kernel.h"

# define INDEX_DATA_TYPE unsigned char
//# define VAL_DATA_TYPE double

#define REPEAT_NUM 3
#define result_check 1
#define AAT 0


typedef struct 
{
	MAT_VAL_TYPE *value;
	int *columnindex;
	MAT_PTR_TYPE *rowpointer;
}SMatrix;

//the number of block nnz
void blocknnz (int *rbnum ,int *cbnum ,  int *csrcolidx , MAT_PTR_TYPE *csrrowptr, int m,int n ,
               int *blknnz )
{

    *rbnum = m%BLOCK_SIZE==0 ? m/BLOCK_SIZE : (m/BLOCK_SIZE)+1 ;

    *cbnum = n%BLOCK_SIZE==0 ? n/BLOCK_SIZE : (n/BLOCK_SIZE)+1 ;

    char *flag=(char *)malloc((*cbnum)*sizeof(char));

    for (int i=0;i< (*rbnum);i++)
	{
        memset(flag,0,(*cbnum) *sizeof(char));
        int start= i*BLOCK_SIZE;
        int end = i== (*rbnum)-1 ?  m : (i+1)*BLOCK_SIZE ;
		for (int j=csrrowptr[start];j<csrrowptr[end];j++){
            int jc=csrcolidx[j]/BLOCK_SIZE;
            if (flag[jc]==0)
            {
                flag[jc]=1;
                (*blknnz)++;
            }
	    } 
	}
}

void blkmessage(int *rbnum ,int *cbnum ,  int *csrcolidx , MAT_PTR_TYPE *csrrowptr, int m,int n ,
                MAT_PTR_TYPE *blkrowptr, int *blkcolIdx)
{
 /*   for (int i=0;i<m +1 ;i++)
    {
        printf("%d     ",csrrowptr[i]);
    }
    printf("\n");
*/
    memset(blkrowptr,0,((*rbnum)+1) *sizeof(MAT_PTR_TYPE));

/*    for(int i=0;i<(*rbnum)+1;i++)
    {
        printf("%d    ",blkrowptr[i]);
    }
    printf("\n");
*/
//    blkcolIdx=(int *)malloc((*blknnz)*sizeof(int));
    char *flag=(char *)malloc((*cbnum)*sizeof(char));

    int colid =0 ;
    for (int i=0;i< (*rbnum);i++)
	{
        memset(flag,0,(*cbnum )*sizeof(char));
        int start= i*BLOCK_SIZE;
        int end = i==(*rbnum)-1 ?  m : (i+1)*BLOCK_SIZE ;
    //    printf("csrrowptr[%d] = %d,csrrowptr[%d] = %d\n",start,csrrowptr[start],end,csrrowptr[end]);
		for (int j=csrrowptr[start];j<csrrowptr[end];j++)
        {
            int jc=csrcolidx[j]/BLOCK_SIZE;
        //    printf("jc =%d\n",jc);
            if (flag[jc]==0)
            {
                flag[jc]=1;
                blkrowptr[i] ++;
                blkcolIdx[colid]=jc;
                colid++;
            }
	    } 
	}

/*    for(int i=0;i<(*rbnum)+1;i++)
    {
        printf("%d    ",blkrowptr[i]);
    }
    printf("\n");
*/
    

    exclusive_scan(blkrowptr,(*rbnum)+1);
}

void step1 (int *d_blkrowptrA, int *d_blkcolidxA, int blkmA, int blknA,
            int *d_blkcolptrB, int *d_blkrowidxB, int blkmB, int blknB,
            int *d_blkrowptrC, int *numblkC) 
{
    struct timeval t1, t2;

    gettimeofday(&t1, NULL);

    memset(d_blkrowptrC,0,(blkmA+1)*sizeof(MAT_PTR_TYPE));

    for (int blki = 0;blki < blkmA ;blki ++)
    {
        int blknnz_C =0;
        for (int blkj =0 ;blkj < blknB ;blkj++)
        {
            int posA = d_blkrowptrA[blki];
            int posB = d_blkcolptrB[blkj];
            int idxA =0;
            int idxB =0;
            int posa_updated = 1;
            int posb_updated = 1;
            int flag =0;
   
            while(posA < d_blkrowptrA[blki + 1] && posB < d_blkcolptrB[blkj + 1])
            {
                
                idxA = posa_updated ? d_blkcolidxA[posA] : idxA ;
                idxB = posb_updated ? d_blkrowidxB[posB] : idxB ;
                if (idxA == idxB)  // do spgemm of this pair
                {
                 //   printf ("do spgemm of this pair, idxa = %i, idxb = %i\n", idxA, idxB);
                //    posA++;
                //    posa_updated = 1;
                //    posB++;
                //    posb_updated = 1;
                    flag =1;
                    break ;
                }
                else 
                {
                //    printf ("the smaller index goes forward (before), posa = %i, posb = %i\n", posA, posA);
                    posA = idxA < idxB ? posA + 1 : posA;
                    posa_updated = idxA < idxB ? 1 : 0;
                    posB = idxA > idxB ? posB + 1 : posB;
                    posb_updated = idxA > idxB ? 1 : 0;
                //    printf ("the smaller index goes forward (after),  posa = %i, posb = %i\n", posA, posA);
                }
            }
            if (flag ==1)
        //    if (posA < d_blkrowptrA[blki + 1] && posB < d_blkcolptrB[blkj + 1])
            {
                blknnz_C ++;
            }
        }
        d_blkrowptrC[blki] = blknnz_C ;
    }

exclusive_scan(d_blkrowptrC,blkmA+1);

*numblkC = d_blkrowptrC[blkmA];

    gettimeofday(&t2, NULL);

    double time_kernel = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("CPU  step1 kernel = %.2f ms, numblkC = %i\n", time_kernel, *numblkC);

}


__forceinline__ __device__
int sum_32_shfl(int sum)
{
    #pragma unroll
    for(int mask = WARP_SIZE / 2 ; mask > 0 ; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    return sum;
}

/*
// inclusive scan
__forceinline__ __device__
int scan_32_shfl(      int x,
                 const int lane_id)
{
    int y = __shfl_up(x, 1);
    x = lane_id >= 1 ? x + y : x;
    y = __shfl_up(x, 2);
    x = lane_id >= 2 ? x + y : x;
    y = __shfl_up(x, 4);
    x = lane_id >= 4 ? x + y : x;
    y = __shfl_up(x, 8);
    x = lane_id >= 8 ? x + y : x;
    y = __shfl_up(x, 16);
    x = lane_id >= 16 ? x + y : x;

    return x;
}
*/

__inline__ __device__
int binary_search_exact_kernel(const int *d_array, int l, int r, int key) 
{ 
    while (l <= r) { 
        int m = l + (r - l) / 2; 
  
        // Check if x is present at mid 
        if (d_array[m] == key) 
            return m; 
  
        // If x greater, ignore left half 
        if (d_array[m] < key) 
            l = m + 1; 
  
        // If x is smaller, ignore right half 
        else
            r = m - 1; 
    } 
  
    // if we reach here, then element was 
    // not present 
    return -1; 
} 

__inline__ __device__
int binary_search_right_boundary_kernel(const int *d_row_pointer,
                                       const int  key_input,
                                       const int  size)
{
    int start = 0;
    int stop  = size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

//#if __CUDA_ARCH__ >= 500
//        key_median = __ldg(&d_row_pointer[median]);
//#else
        key_median = d_row_pointer[median];
//#endif

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start - 1;
}

int binary_search_right_boundary_kernel_cpu(const int *d_row_pointer,
                                       const int  key_input,
                                       const int  size)
{
    int start = 0;
    int stop  = size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

//#if __CUDA_ARCH__ >= 500
//        key_median = __ldg(&d_row_pointer[median]);
//#else
        key_median = d_row_pointer[median];
//#endif

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start - 1;
}


// step 1. get the number of non-empty blocks in each block-row of C
//         #thread-blocks = #block-row of C
__global__
void stir_spgemm_step1_cuda_kernel(int *d_blkrowptrA, int *d_blkcolidxA, int blkmA, 
                                   int *d_blkcolptrB, int *d_blkrowidxB, int blknB,
                                   int *d_blkrowptrC)
{
    int blki = blockIdx.x;
    //int local_warp_id = threadIdx.x / WARP_SIZE;

    const int abase = d_blkrowptrA[blki];
    const int astop = d_blkrowptrA[blki+1];
    const int astart = d_blkcolidxA[abase];
    const int aend = d_blkcolidxA[astop-1];
    const int lena = astop - abase;

    __shared__ int s_blkcolidxA[128];
    __shared__ int s_numblkC[1];

    if (lena <= 128)
        for (int i = threadIdx.x; i < lena; i += blockDim.x)
            s_blkcolidxA[i] = d_blkcolidxA[abase + i];

    if (!threadIdx.x) 
        s_numblkC[0] = 0;
    __syncthreads();

    for (int blkj = threadIdx.x; blkj < blknB; blkj += blockDim.x)
    {
        const int bbase = d_blkcolptrB[blkj];
        const int bstop = d_blkcolptrB[blkj+1];
        const int bstart = d_blkrowidxB[bbase];
        const int bend = d_blkrowidxB[bstop-1];
        const int lenb = bstop - bbase;

        if (aend < bstart || bend < astart)
            continue;

        int posa_real = 0;
        int posb_real = 0;
        if (bstart > astart)
        {
            posa_real = binary_search_right_boundary_kernel((lena <= 128 ? s_blkcolidxA : &d_blkcolidxA[abase]), bstart, lena);
        }
        else if (bstart < astart)
        {
            posb_real = binary_search_right_boundary_kernel(&d_blkrowidxB[bbase], astart, lenb);
        }
        /*else if (bstart == astart)
        {
            atomicAdd(&s_numblkC[0], 1);
            //atomicAdd(&d_blkrowptrC[blki], 1);
            continue;
        }*/


//printf("blki = %i, blkj = %i, astart = %i, aend = %i, bstart = %i, bend = %i\n",
//        blki, blkj, astart, aend, bstart, bend);

        // first check if it is possible to intersect
        // if not, do nothing and continue
        /*if (aend == bstart)
        {
            //atomicAdd(&numblkC[0], 1);
            atomicAdd(&d_blkrowptrC[blki], 1);
            //continue;
        }
        else if (aend < bstart)*/

        //if (aend < bstart || bend < astart)
        //    continue;

        //if (aend >= bstart && bend >= astart)
        {
            int posa = posa_real;
            int posb = posb_real;
            int idxa = 0;
            int idxb = 0;
            int posa_updated = 1;
            int posb_updated = 1;

            while(posa < lena && posb < lenb)
            {
                //idxa = posa_updated ? d_blkcolidxA[abase+posa] : idxa; //a[posa] : idxa;
                idxa = posa_updated ? (lena < 128 ? s_blkcolidxA[posa] : d_blkcolidxA[abase+posa]) : idxa; //a[posa] : idxa;
                idxb = posb_updated ? d_blkrowidxB[bbase+posb] : idxb; //b[posb] : idxb;

                if (idxa == idxb)
                {
                    atomicAdd(&s_numblkC[0], 1);
                    //atomicAdd(&d_blkrowptrC[blki], 1);
                    break;

                    // do spgemm of this pair
                    //printf ("do spgemm of this pair, idxa = %i, idxb = %i\n", idxa, idxb);
                    //posa++;
                    //posa_updated = 1;
                    //posb++;
                    //posb_updated = 1;
                }
                else
                {
                    // the smaller index goes forward
                    //printf ("the smaller index goes forward (before), posa = %i, posb = %i\n", posa, posb);
            //posa = idxa < idxb ? posa + 1 : posa;
            posa_updated = idxa < idxb ? 1 : 0;
            posa += posa_updated;
            //posb = idxa > idxb ? posb + 1 : posb;
            posb_updated = idxa > idxb ? 1 : 0;
            posb += posb_updated;
                    //printf ("the smaller index goes forward (after),  posa = %i, posb = %i\n", posa, posb);
                }
            }
        }
    }
    __syncthreads();

    if (!threadIdx.x)
        d_blkrowptrC[blki] = s_numblkC[0];
}


void step1_cuda (int *blkrowptrA, int *blkcolidxA, int blkmA, int blknA, int numblkA,
            int *blkcolptrB, int *blkrowidxB, int blkmB, int blknB, int numblkB,
            int *blkrowptrC_golden, int *numblkC) 
{
//for (int i = 0; i < blkmA; i++)
//    printf("[%i],%i\n", i, blkrowptrA[i+1] - blkrowptrA[i]);

int *d_blkrowptrA;
 int *d_blkcolidxA; 
            int *d_blkcolptrB; int *d_blkrowidxB;
            int *d_blkrowptrC;

    cudaMalloc((void **)&d_blkrowptrA, (blkmA+1) * sizeof(int));
    cudaMalloc((void **)&d_blkcolidxA, numblkA * sizeof(int));
    cudaMalloc((void **)&d_blkcolptrB, (blknB+1) * sizeof(int));
    cudaMalloc((void **)&d_blkrowidxB, numblkB * sizeof(int));

    cudaMalloc((void **)&d_blkrowptrC, (blkmA+1) * sizeof(int));
    cudaMemset(d_blkrowptrC, 0, (blkmA+1) * sizeof(int));

    cudaMemcpy(d_blkrowptrA,     blkrowptrA,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcolidxA,     blkcolidxA,     numblkA * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcolptrB,     blkcolptrB,     (blknB+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkrowidxB,     blkrowidxB,     numblkB * sizeof(int),              cudaMemcpyHostToDevice);

    // call cuda kernel
    struct timeval t1, t2;

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

    int num_threads = 64;
    int num_blocks = blkmA;
    stir_spgemm_step1_cuda_kernel<<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, blkmA, 
                              d_blkcolptrB, d_blkrowidxB, blknB, d_blkrowptrC);

exclusive_scan_device_cuda(d_blkrowptrC, blkmA+1);
int nbc = 0;
cudaMemcpy(&nbc,     &d_blkrowptrC[blkmA],     sizeof(int), cudaMemcpyDeviceToHost);
*numblkC = nbc;

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    double time_kernel = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("CUDA step1 kernel = %.2f ms, numblkC = %i\n", time_kernel, nbc);

    int *h_blkrowptrC = (int *)malloc((blkmA+1)*sizeof(int));
    cudaMemcpy(h_blkrowptrC,     d_blkrowptrC,     (blkmA+1) * sizeof(int), cudaMemcpyDeviceToHost);

    int errcnt = 0;
    for (int i = 0; i < (blkmA+1); i++)
        if (h_blkrowptrC[i] != blkrowptrC_golden[i])
            errcnt++;
    printf("step 1, blkrowptrC, errcnt = %i\n", errcnt);

    cudaFree(d_blkrowptrA);
    cudaFree(d_blkcolidxA);
    cudaFree(d_blkcolptrB);
    cudaFree(d_blkrowidxB);
    cudaFree(d_blkrowptrC);
    free(h_blkrowptrC);
}


void step1_cuda_new (sfBIN *bin, int *blkrowptrA, int *blkcolidxA, int blkmA, int blknA, int numblkA,
            int *blkrowptrB, int *blkcolidxB, int blkmB, int blknB, int numblkB,
            int *blkrowptrC_golden, int *numblkC) 
{
//for (int i = 0; i < blkmA; i++)
//    printf("[%i],%i\n", i, blkrowptrA[i+1] - blkrowptrA[i]);

int *d_blkrowptrA;
 int *d_blkcolidxA; 
            int *d_blkrowptrB; int *d_blkcolidxB;
            int *d_blkrowptrC;

    cudaMalloc((void **)&d_blkrowptrA, (blkmA+1) * sizeof(int));
    cudaMalloc((void **)&d_blkcolidxA, numblkA * sizeof(int));
    cudaMalloc((void **)&d_blkrowptrB, (blkmB+1) * sizeof(int));
    cudaMalloc((void **)&d_blkcolidxB, numblkB * sizeof(int));

    cudaMalloc((void **)&d_blkrowptrC, (blkmA+1) * sizeof(int));
    cudaMemset(d_blkrowptrC, 0, (blkmA+1) * sizeof(int));

    cudaMemcpy(d_blkrowptrA,     blkrowptrA,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcolidxA,     blkcolidxA,     numblkA * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkrowptrB,     blkrowptrB,     (blkmB+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcolidxB,     blkcolidxB,     numblkB * sizeof(int),              cudaMemcpyHostToDevice);

    // call cuda kernel
    struct timeval t1, t2;

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

/*
    int num_threads = 64;
    int num_blocks = blkmA;
    stir_spgemm_step1_cuda_kernel<<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, blkmA, 
                              d_blkcolptrB, d_blkrowidxB, blknB, d_blkrowptrC);

exclusive_scan_device_cuda(d_blkrowptrC, blkmA+1);
int nbc = 0;
cudaMemcpy(&nbc,     &d_blkrowptrC[blkmA],     sizeof(int), cudaMemcpyDeviceToHost);
*numblkC = nbc;
*/

    /* Initialize bin */
    init_bin(bin, blkmA);

    /* Set max bin */
    //set_max_bin(a->d_rpt, a->d_col, b->d_rpt, &bin, M);
    set_max_bin(d_blkrowptrA, d_blkcolidxA, d_blkrowptrB, bin, blkmA);

    //    cudaMalloc((void **)&d_csrRowPtrC, (blkmA+1) * sizeof(int));

    /* Count nz of C */
    set_row_nnz(d_blkrowptrA, d_blkcolidxA,
                d_blkrowptrB, d_blkcolidxB,
                d_blkrowptrC,
                bin,
                blkmA,
                numblkC);

   //printf("nsparse nnzC = %i\n", nnzC);

    /* Set bin */
    set_min_bin(bin, blkmA);

        //spgemm_nsparse_executor_step1(mA, nA,  nnzA, d_csrRowPtrA, d_csrColIdxA, 
        //                              mB, nB,  nnzB, d_csrRowPtrB, d_csrColIdxB, 
        //                              mC, nC, &nnzC, d_csrRowPtrC);

     //   cudaMalloc((void **)&d_csrColIdxC, *numblkC * sizeof(int));

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    double time_kernel = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("CUDA step1 new kernel = %.2f ms, numblkC = %i\n", time_kernel, *numblkC);

    int *h_blkrowptrC = (int *)malloc((blkmA+1)*sizeof(int));
    cudaMemcpy(h_blkrowptrC,     d_blkrowptrC,     (blkmA+1) * sizeof(int), cudaMemcpyDeviceToHost);

    int errcnt = 0;
    for (int i = 0; i < (blkmA+1); i++)
        if (h_blkrowptrC[i] != blkrowptrC_golden[i]){ //printf("%i, %i\n", h_blkrowptrC[i], blkrowptrC_golden[i]);
            errcnt++;}
    printf("step 1 new, blkrowptrC, errcnt = %i\n", errcnt);

    cudaFree(d_blkrowptrA);
    cudaFree(d_blkcolidxA);
    cudaFree(d_blkrowptrB);
    cudaFree(d_blkcolidxB);
    cudaFree(d_blkrowptrC);
    free(h_blkrowptrC);
}



void step2 (int *d_blkrowptrA, int *d_blkcolidxA, int blkmA, int blknA,
            int *d_blkcolptrB, int *d_blkrowidxB, int blkmB, int blknB,
            int *d_blkrowptrC, int *d_blkcolidxC) 
{
    struct timeval t1, t2;

    gettimeofday(&t1, NULL);
//    memset(d_blkrowptrC,0,(blkmA+1)*sizeof(MAT_PTR_TYPE));

    int blkcolcount =0;
    for (int blki = 0;blki < blkmA ;blki ++)
    {
        int blknnz_C =0;
        for (int blkj =0 ;blkj < blknB ;blkj++)
        {
            int posA = d_blkrowptrA[blki];
            int posB = d_blkcolptrB[blkj];
            int idxA =0;
            int idxB =0;
            int posa_updated = 1;
            int posb_updated = 1;
        //    int flag =0;
            while(posA < d_blkrowptrA[blki + 1] && posB < d_blkcolptrB[blkj + 1])
            {
                
                idxA = posa_updated ? d_blkcolidxA[posA] : idxA ;
                idxB = posb_updated ? d_blkrowidxB[posB] : idxB ;
                if (idxA == idxB)  // do spgemm of this pair
                {
                //    printf ("do spgemm of this pair, idxa = %i, idxb = %i\n", idxA, idxB);
                    d_blkcolidxC[blkcolcount] = blkj;
                    blkcolcount ++;
                //    posA++;
                //    posa_updated = 1;
                //    posB++;
                //    posb_updated = 1;
                //    flag =1;
                    break ;
                }
                else 
                {
                //    printf ("the smaller index goes forward (before), posa = %i, posb = %i\n", posA, posA);
                    posA = idxA < idxB ? posA + 1 : posA;
                    posa_updated = idxA < idxB ? 1 : 0;
                    posB = idxA > idxB ? posB + 1 : posB;
                    posb_updated = idxA > idxB ? 1 : 0;
                //    printf ("the smaller index goes forward (after),  posa = %i, posb = %i\n", posA, posA);
                }
            }
        }
    }
        gettimeofday(&t2, NULL);

    double time_kernel = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("CPU  step2 kernel = %.2f ms\n", time_kernel);

}


// step 2. get the blkcolidx of the non-empty blocks found in each block-row of C
__global__
void stir_spgemm_step2_cuda_kernel(int *d_blkrowptrA, int *d_blkcolidxA, int blkmA, 
                                   int *d_blkcolptrB, int *d_blkrowidxB, int blknB,
                                   int *d_blkrowptrC_offset, int *d_blkrowidxC, int *d_blkcolidxC)
{
    int blki = blockIdx.x;
    //int local_warp_id = threadIdx.x / WARP_SIZE;

    const int abase = d_blkrowptrA[blki];
    const int astop = d_blkrowptrA[blki+1];
    const int astart = d_blkcolidxA[abase];
    const int aend = d_blkcolidxA[astop-1];
    const int lena = astop - abase;

    __shared__ int s_blkcolidxA[128];
    __shared__ int s_numblkC[1];

    if (lena <= 128)
        for (int i = threadIdx.x; i < lena; i += blockDim.x)
            s_blkcolidxA[i] = d_blkcolidxA[abase + i];

   // if (!threadIdx.x) 
    //    s_numblkC[0] = 0;
    __syncthreads();

    for (int blkj = threadIdx.x; blkj < blknB; blkj += blockDim.x)
    {
        const int bbase = d_blkcolptrB[blkj];
        const int bstop = d_blkcolptrB[blkj+1];
        const int bstart = d_blkrowidxB[bbase];
        const int bend = d_blkrowidxB[bstop-1];
        const int lenb = bstop - bbase;

        if (aend < bstart || bend < astart)
            continue;

        int posa_real = 0;
        int posb_real = 0;
        if (bstart > astart)
        {
            posa_real = binary_search_right_boundary_kernel((lena <= 128 ? s_blkcolidxA : &d_blkcolidxA[abase]), bstart, lena);
        }
        else if (bstart < astart)
        {
            posb_real = binary_search_right_boundary_kernel(&d_blkrowidxB[bbase], astart, lenb);
        }
        /*else if (bstart == astart)
        {
            atomicAdd(&s_numblkC[0], 1);
            //atomicAdd(&d_blkrowptrC[blki], 1);
            continue;
        }*/


//printf("blki = %i, blkj = %i, astart = %i, aend = %i, bstart = %i, bend = %i\n",
//        blki, blkj, astart, aend, bstart, bend);

        // first check if it is possible to intersect
        // if not, do nothing and continue
        /*if (aend == bstart)
        {
            //atomicAdd(&numblkC[0], 1);
            atomicAdd(&d_blkrowptrC[blki], 1);
            //continue;
        }
        else if (aend < bstart)*/

        //if (aend < bstart || bend < astart)
        //    continue;

        //if (aend >= bstart && bend >= astart)
        {
            int posa = posa_real;
            int posb = posb_real;
            int idxa = 0;
            int idxb = 0;
            int posa_updated = 1;
            int posb_updated = 1;

            while(posa < lena && posb < lenb)
            {
                //idxa = posa_updated ? d_blkcolidxA[abase+posa] : idxa; //a[posa] : idxa;
                idxa = posa_updated ? (lena < 128 ? s_blkcolidxA[posa] : d_blkcolidxA[abase+posa]) : idxa; //a[posa] : idxa;
                idxb = posb_updated ? d_blkrowidxB[bbase+posb] : idxb; //b[posb] : idxb;

                if (idxa == idxb)
                {
                    //atomicAdd(&s_numblkC[0], 1);
                    int posc = atomicAdd(&d_blkrowptrC_offset[blki], 1);
                    d_blkrowidxC[posc] = blki;
                    d_blkcolidxC[posc] = blkj;
                    //atomicAdd(&d_blkrowptrC[blki], 1);
                    break;

                    // do spgemm of this pair
                    //printf ("do spgemm of this pair, idxa = %i, idxb = %i\n", idxa, idxb);
                    //posa++;
                    //posa_updated = 1;
                    //posb++;
                    //posb_updated = 1;
                }
                else
                {
                    // the smaller index goes forward
                    //printf ("the smaller index goes forward (before), posa = %i, posb = %i\n", posa, posb);
            //posa = idxa < idxb ? posa + 1 : posa;
            posa_updated = idxa < idxb ? 1 : 0;
            posa += posa_updated;
            //posb = idxa > idxb ? posb + 1 : posb;
            posb_updated = idxa > idxb ? 1 : 0;
            posb += posb_updated;
                    //printf ("the smaller index goes forward (after),  posa = %i, posb = %i\n", posa, posb);
                }
            }
        }
    }
    //__syncthreads();

    //if (!threadIdx.x)
    //    d_blkrowptrC[blki] = s_numblkC[0];
}

void step2_cuda (int *blkrowptrA, int *blkcolidxA, int blkmA, int blknA, int numblkA,
            int *blkcolptrB, int *blkrowidxB, int blkmB, int blknB, int numblkB,
            int *blkrowptrC, int *blkcolidxC_golden, int *d_blkrowidxC, int *d_blkcolidxC, int numblkC) 
{
//for (int i = 0; i < blkmA; i++)
//    printf("[%i],%i\n", i, blkrowptrA[i+1] - blkrowptrA[i]);

int *d_blkrowptrA;
 int *d_blkcolidxA; 
            int *d_blkcolptrB; int *d_blkrowidxB;
            int *d_blkrowptrC;
            int *d_blkrowptrC_offset;

    cudaMalloc((void **)&d_blkrowptrA, (blkmA+1) * sizeof(int));
    cudaMalloc((void **)&d_blkcolidxA, numblkA * sizeof(int));
    cudaMalloc((void **)&d_blkcolptrB, (blkmB+1) * sizeof(int));
    cudaMalloc((void **)&d_blkrowidxB, numblkB * sizeof(int));
    cudaMalloc((void **)&d_blkrowptrC, (blkmA+1) * sizeof(int));
    cudaMalloc((void **)&d_blkrowptrC_offset, (blkmA+1) * sizeof(int));

    cudaMemcpy(d_blkrowptrA,     blkrowptrA,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcolidxA,     blkcolidxA,     numblkA * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcolptrB,     blkcolptrB,     (blkmB+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkrowidxB,     blkrowidxB,     numblkB * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkrowptrC,     blkrowptrC,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkrowptrC_offset,     d_blkrowptrC,     (blkmA+1) * sizeof(int),              cudaMemcpyDeviceToDevice);

    // call cuda kernel
    struct timeval t1, t2;

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

    int num_threads = 64;
    int num_blocks = blkmA;
    stir_spgemm_step2_cuda_kernel<<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, blkmA, 
                              d_blkcolptrB, d_blkrowidxB, blknB, 
                              d_blkrowptrC, d_blkrowidxC, d_blkcolidxC);

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    double time_kernel = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("CUDA step2 kernel = %.2f ms\n", time_kernel);

    int *h_blkcolidxC = (int *)malloc(numblkC*sizeof(int));
    cudaMemcpy(h_blkcolidxC,     d_blkcolidxC,     numblkC * sizeof(int), cudaMemcpyDeviceToHost);

    for(int blki =0;blki < blkmA ;blki ++)
    {
        quick_sort_key(h_blkcolidxC + blkrowptrC[blki],blkrowptrC[blki+1] - blkrowptrC[blki]);
    }

    cudaMemcpy(d_blkcolidxC,     h_blkcolidxC,     numblkC * sizeof(int), cudaMemcpyHostToDevice);

    int errcnt = 0;
    for (int i = 0; i < numblkC; i++)
        if (h_blkcolidxC[i] != blkcolidxC_golden[i])
            errcnt++;
    printf("step 2, h_blkcolidxC, errcnt = %i\n", errcnt);

    cudaFree(d_blkrowptrA);
    cudaFree(d_blkcolidxA);
    cudaFree(d_blkcolptrB);
    cudaFree(d_blkrowidxB);
    cudaFree(d_blkrowptrC);
    cudaFree(d_blkrowptrC_offset);
    free(h_blkcolidxC);
}


void step2_cuda_new (sfBIN *bin, int *blkrowptrA, int *blkcolidxA, int blkmA, int blknA, int numblkA,
            int *blkrowptrB, int *blkcolidxB, int blkmB, int blknB, int numblkB,
            int *blkrowptrC, int *blkcolidxC_golden, int *d_blkrowidxC, int *d_blkcolidxC, int numblkC) 
{
//for (int i = 0; i < blkmA; i++)
//    printf("[%i],%i\n", i, blkrowptrA[i+1] - blkrowptrA[i]);

int *d_blkrowptrA;
 int *d_blkcolidxA; 
            int *d_blkrowptrB; int *d_blkcolidxB;
            int *d_blkrowptrC;
            int *d_blkrowptrC_offset;

    cudaMalloc((void **)&d_blkrowptrA, (blkmA+1) * sizeof(int));
    cudaMalloc((void **)&d_blkcolidxA, numblkA * sizeof(int));
    cudaMalloc((void **)&d_blkrowptrB, (blkmB+1) * sizeof(int));
    cudaMalloc((void **)&d_blkcolidxB, numblkB * sizeof(int));
    cudaMalloc((void **)&d_blkrowptrC, (blkmA+1) * sizeof(int));
    cudaMalloc((void **)&d_blkrowptrC_offset, (blkmA+1) * sizeof(int));

    cudaMemcpy(d_blkrowptrA,     blkrowptrA,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcolidxA,     blkcolidxA,     numblkA * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkrowptrB,     blkrowptrB,     (blkmB+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcolidxB,     blkcolidxB,     numblkB * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkrowptrC,     blkrowptrC,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkrowptrC_offset,     d_blkrowptrC,     (blkmA+1) * sizeof(int),              cudaMemcpyDeviceToDevice);

    // call cuda kernel
    struct timeval t1, t2;

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
/*
    int num_threads = 64;
    int num_blocks = blkmA;
    stir_spgemm_step2_cuda_kernel<<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, blkmA, 
                              d_blkcolptrB, d_blkrowidxB, blknB, 
                              d_blkrowptrC, d_blkrowidxC, d_blkcolidxC);
*/

    /* Calculating value of C */
    calculate_value_col_bin(d_blkrowptrA, d_blkcolidxA, NULL,
                            d_blkrowptrB, d_blkcolidxB, NULL,
                            d_blkrowptrC, d_blkrowidxC, d_blkcolidxC, NULL,
                            bin,
                            blkmA, blkmB);

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    double time_kernel = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("CUDA step2 new kernel = %.2f ms\n", time_kernel);

   //int *h_blkrowidxC = (int *)malloc(numblkC*sizeof(int));
    //cudaMemcpy(h_blkrowidxC,     d_blkrowidxC,     numblkC * sizeof(int), cudaMemcpyDeviceToHost);


    int *h_blkcolidxC = (int *)malloc(numblkC*sizeof(int));
    cudaMemcpy(h_blkcolidxC,     d_blkcolidxC,     numblkC * sizeof(int), cudaMemcpyDeviceToHost);

    for(int blki =0;blki < blkmA ;blki ++)
    {
        quick_sort_key(h_blkcolidxC + blkrowptrC[blki],blkrowptrC[blki+1] - blkrowptrC[blki]);
    }

    cudaMemcpy(d_blkcolidxC,     h_blkcolidxC,     numblkC * sizeof(int), cudaMemcpyHostToDevice);

    //for(int blki =0;blki < blkmA ;blki ++)
    //{
    //    quick_sort_key(h_blkcolidxC + blkrowptrC[blki],blkrowptrC[blki+1] - blkrowptrC[blki]);
    //}

    //cudaMemcpy(d_blkcolidxC,     h_blkcolidxC,     numblkC * sizeof(int), cudaMemcpyHostToDevice);

    int errcnt = 0;
    for (int i = 0; i < numblkC; i++)
        if (h_blkcolidxC[i] != blkcolidxC_golden[i])
            {//if (h_blkrowidxC[i] == 7) printf("[%i] (%i %i) %i\n", i, h_blkrowidxC[i], h_blkcolidxC[i], blkcolidxC_golden[i]);
            errcnt++;}
    printf("step 2 new, h_blkcolidxC, errcnt = %i\n", errcnt);

    cudaFree(d_blkrowptrA);
    cudaFree(d_blkcolidxA);
    cudaFree(d_blkrowptrB);
    cudaFree(d_blkcolidxB);
    cudaFree(d_blkrowptrC);
    cudaFree(d_blkrowptrC_offset);
    free(h_blkcolidxC);
}


void step3 (int *d_blkrowptrA, int *d_blkcolidxA, int blkmA, int blknA,int *nnzb_A ,int mA,
            MAT_VAL_TYPE *blkcsr_Val_A , unsigned char  *blkcsr_Col_A , unsigned char *blkcsr_Ptr_A ,
            int *d_blkcolptrB, int *d_blkrowidxB, int blkmB, int blknB , int *nnzb_B ,
            MAT_VAL_TYPE *blkcsr_Val_B , unsigned char  *blkcsr_Col_B , unsigned char *blkcsr_Ptr_B ,
            int *d_blkrowptrC, int *d_blkcolidxC,int *nnzb_C, int *nnzC)
{

    struct timeval t1, t2;

    gettimeofday(&t1, NULL);
/*    for (int i=0;i<d_blkrowptrA[blkmA];i++)
    {
        printf("%d     ",d_blkcolidxA[i]);
    }
    printf("\n");

    for (int i=0;i<d_blkcolptrB[blknB];i++)
    {
        printf("%d     ",d_blkrowidxB[i]);
    }
    printf("\n");
*/
    char * blkc = (char *)malloc((BLOCK_SIZE * BLOCK_SIZE) *sizeof(char));

    for (int blki =0 ;blki <blkmA ;blki++)
    {
        int rowlen = blki == blkmA -1 ? mA- (blkmA -1 ) * BLOCK_SIZE : BLOCK_SIZE ;
        for (int blkj =d_blkrowptrC[blki]; blkj <d_blkrowptrC[blki + 1]; blkj++)
        {
            int count =0 ;
            int blkccolid = d_blkcolidxC[blkj];
        //    int rowlen = blki == blkmA -1 ? mA- (blkmA -1 ) * BLOCK_SIZE : BLOCK_SIZE ;
        //    int collen = blkccolid == blknB -1 ? nB - (blknB -1) *BLOCK_SIZE : BLOCK_SIZE ;
            memset (blkc , 0, (BLOCK_SIZE * BLOCK_SIZE) *sizeof(char));

            int posA = d_blkrowptrA[blki];
            int posB = d_blkcolptrB[blkccolid];
            int idxA= 0;
            int idxB =0;
            int posa_updated =1;
            int posb_updated =1;
            while (posA < d_blkrowptrA[blki +1] && posB <d_blkcolptrB[blkccolid + 1])
            {
                idxA = posa_updated ? d_blkcolidxA[posA] : idxA ;
                idxB = posb_updated ? d_blkrowidxB[posB] : idxB ;
                if (idxA == idxB)  // do spgemm of this pair
                {
                //        printf ("do spgemm of this pair, idxa = %i, idxb = %i\n", idxA, idxB);
                //for each row of block
                    for (int ri =0;ri <BLOCK_SIZE ;ri ++ )
                    {
                        if (ri == rowlen)
                            break;
                        int stopa = ri == BLOCK_SIZE -1 ? nnzb_A[posA +1] - nnzb_A[posA] : blkcsr_Ptr_A[posA * BLOCK_SIZE + ri + 1] ;
                
                        for (int i=blkcsr_Ptr_A[ posA * BLOCK_SIZE+ ri];i<stopa;i++)
                        {
                            int cola= blkcsr_Col_A[nnzb_A[posA]+i] ;
                            int stopb = cola == BLOCK_SIZE -1  ? nnzb_B[posB +1]- nnzb_B[posB] : blkcsr_Ptr_B[posB * BLOCK_SIZE+cola +1] ;
                            for (int bi= blkcsr_Ptr_B[posB * BLOCK_SIZE +cola ];bi< stopb; bi++)
                            {
                                const int colb = blkcsr_Col_B[nnzb_B[posB] + bi];
                                if (blkc[ri * BLOCK_SIZE + colb] == 0)
                                {
                                    blkc[ri * BLOCK_SIZE + colb] = 1;
                                }
                            }

                        }
                    }
                    posA++;
                    posa_updated = 1;
                    posB++;
                    posb_updated = 1;
                }
                else 
                {
                //    printf ("the smaller index goes forward (before), posa = %i, posb = %i\n", posA, posA);
                    posA = idxA < idxB ? posA + 1 : posA;
                    posa_updated = idxA < idxB ? 1 : 0;
                    posB = idxA > idxB ? posB + 1 : posB;
                    posb_updated = idxA > idxB ? 1 : 0;
                //    printf ("the smaller index goes forward (after),  posa = %i, posb = %i\n", posA, posA);
                }

            }  
            for (int ci=0;ci< BLOCK_SIZE * BLOCK_SIZE ; ci ++)
            {
                if (blkc[ci]== 1)
                {
                    count ++ ;
                }
            }
            nnzb_C[blkj] = count ;
        //    printf("count = %d\n",count);
        }
    }

    exclusive_scan(nnzb_C,d_blkrowptrC[blkmA] + 1);

    *nnzC = nnzb_C[d_blkrowptrC[blkmA]];

        gettimeofday(&t2, NULL);

    double time_kernel = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("CPU  step3 kernel = %.2f ms\n", time_kernel);
}

template <int SMEM_SIZE>
__global__
void stir_spgemm_step3_cuda_kernel
                             (int *d_blkrowptrA,
                            const int* __restrict__ d_blkcolidxA,
                            int *d_nnzb_A,
                            MAT_VAL_TYPE *d_blkcsr_Val_A,
                            unsigned char *d_blkcsr_Col_A,
                            unsigned char *d_blkcsr_Ptr_A,
                            unsigned short *d_blkmaskA,
                            int blkmA, int blknA, int numblkA, int nnzA, 
                            int *d_blkcolptrB,
                            const int* __restrict__ d_blkrowidxB, 
                            int *d_nnzb_B,
                            MAT_VAL_TYPE *d_blkcsr_Val_B,
                            unsigned char *d_blkcsr_Col_B,
                            unsigned char *d_blkcsr_Ptr_B,
                            unsigned short *d_blkmaskB,
                            int blkmB, int blknB , int numblkB, int nnzB, 
                            int *d_blkrowidxC,
                            int *d_blkcolidxC,
                            unsigned char *d_blkcsr_Ptr_C,
                            int *d_nnzb_C,
                            unsigned short *d_blkmaskC,
                            int *d_blksmem_cnt,
                            int *d_blkdns_cnt,
                            int *d_blkid_smem,
                            int *d_blkid_dns,
                            int numblkC)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp_id = global_id / WARP_SIZE;

    if (global_warp_id >= numblkC) return;
    //if (global_warp_id < numblkC)

    __shared__ unsigned int s_maskc[WARP_PER_BLOCK * BLOCK_SIZE];
    //__shared__ unsigned short s_blkmaskA[WARP_PER_BLOCK * BLOCK_SIZE];
    __shared__ unsigned short s_blkmaskB[WARP_PER_BLOCK * BLOCK_SIZE];

    __shared__ int s_matched_posa[WARP_PER_BLOCK * SMEM_SIZE];
    __shared__ int s_matched_posb[WARP_PER_BLOCK * SMEM_SIZE];
    __shared__ int s_matchedcnt[WARP_PER_BLOCK];

    //const int local_warp_id = threadIdx.x / WARP_SIZE;
    //__shared__ char s_dnsidx[WARP_PER_BLOCK * BLOCK_SIZE * BLOCK_SIZE];
    //char *s_dnsidx_local = &s_dnsidx[local_warp_id * BLOCK_SIZE * BLOCK_SIZE];

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    //const int sublane_id = lane_id % (WARP_SIZE / BLOCK_SIZE);
    //const int subwarp_id = lane_id / (WARP_SIZE / BLOCK_SIZE);
    unsigned int maskc = 0;
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    
    unsigned int *s_maskc_local = &s_maskc[local_warp_id * BLOCK_SIZE];
    //unsigned short *s_blkmaskA_local = &s_blkmaskA[local_warp_id * BLOCK_SIZE];
    unsigned short *s_blkmaskB_local = &s_blkmaskB[local_warp_id * BLOCK_SIZE];

    int *s_matched_posa_local = &s_matched_posa[local_warp_id * SMEM_SIZE];
    int *s_matched_posb_local = &s_matched_posb[local_warp_id * SMEM_SIZE];
    int *s_matchedcnt_local = &s_matchedcnt[local_warp_id];

    //for (int i = 0; i < BLOCK_SIZE; i++)
    //    if (lane_id < BLOCK_SIZE)
    //        s_dnsidx_local[i * BLOCK_SIZE + lane_id] = 0;
    //__syncthreads();
    if (lane_id < BLOCK_SIZE)
        s_maskc_local[lane_id] = 0;
    if (!lane_id) 
        s_matchedcnt_local[0] = 0;
    //__syncthreads();

    const int blki = d_blkrowidxC[global_warp_id];
    const int blkj = d_blkcolidxC[global_warp_id];

    const int abase = d_blkrowptrA[blki];
    const int astop = d_blkrowptrA[blki+1];
    const int astart = d_blkcolidxA[abase];
    const int aend = d_blkcolidxA[astop-1];
    int lena = d_blkrowptrA[blki+1] - abase;

    const int bbase = d_blkcolptrB[blkj];
    const int bstop = d_blkcolptrB[blkj+1];
    const int bstart = d_blkrowidxB[bbase];
    const int bend = d_blkrowidxB[bstop-1];
    int lenb = d_blkcolptrB[blkj+1] - bbase;


    if (lena < lenb)
    {
        for (int i = lane_id; i < lena; i+= WARP_SIZE)
        {
            int idxa = d_blkcolidxA[abase+i];
            int res = binary_search_exact_kernel(&d_blkrowidxB[bbase], 0, lenb-1, idxa);
            if (res != -1)
            {
                int pos = atomicAdd(s_matchedcnt_local, 1);
                if (pos < SMEM_SIZE)
                {
                    s_matched_posa_local[pos] = i;
                    s_matched_posb_local[pos] = res;
                }
            }
        }
    }
    else 
    {
        for (int i = lane_id; i < lenb; i+= WARP_SIZE)
        {
            int idxb = d_blkrowidxB[bbase+i];
            int res = binary_search_exact_kernel(&d_blkcolidxA[abase], 0, lena-1, idxb);
            if (res != -1)
            {
                int pos = atomicAdd(s_matchedcnt_local, 1);
                if (pos < SMEM_SIZE)
                {
                    s_matched_posa_local[pos] = res;
                    s_matched_posb_local[pos] = i;
                }
            }
        }
    }

    int matchedcnt = s_matchedcnt_local[0];

    if (matchedcnt <= SMEM_SIZE)
    {
        for (int i = 0; i < matchedcnt; i++)
        {
            int posa = s_matched_posa_local[i];
            int posb = s_matched_posb_local[i];

                if (lane_id < BLOCK_SIZE)
                {
                    s_blkmaskB_local[lane_id] = d_blkmaskB[(bbase+posb) * BLOCK_SIZE + lane_id];
                }

                const int nnzastart = d_nnzb_A[(abase+posa)];
                int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;

                for (int i = lane_id; i < nnztotala; i+= WARP_SIZE)
                {
                    unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+i];
                    unsigned int maskb = s_blkmaskB_local[rowcolidx & 0xf];
                    atomicOr(&s_maskc_local[rowcolidx >> 4], maskb);
                }
        }
    }
    else 
    {
        int posa_real = 0;
        int posb_real = 0;

        if (bstart > astart)
        {
            //posa_real = binary_search_right_boundary_kernel((lena <= 128 ? s_blkcolidxA : &d_blkcolidxA[abase]), bstart, lena);
            //posa_real = binary_search_right_boundary_kernel(&d_blkcolidxA[abase], bstart, lena);
            int posa_real_new = binary_search_right_boundary_kernel(&d_blkcolidxA[abase], bstart, lena);
            posa_real = posa_real_new < 0 ? 0 : posa_real_new;
        }
        else if (bstart < astart)
        {
            //posb_real = binary_search_right_boundary_kernel(&d_blkrowidxB[bbase], astart, lenb);
            int posb_real_new = binary_search_right_boundary_kernel(&d_blkrowidxB[bbase], astart, lenb);
            posb_real = posb_real_new < 0 ? 0 : posb_real_new;
        }

        if (bstop < astop)
        {
            int lena_new = binary_search_right_boundary_kernel(&d_blkcolidxA[abase], bend, lena) + 1;
            lena = lena_new > lena ? lena : lena_new;
        }
        else if (bstop > astop)
        {
            int lenb_new = binary_search_right_boundary_kernel(&d_blkrowidxB[bbase], aend, lenb) + 1;
            lenb = lenb_new > lenb ? lenb : lenb_new;
        }
        for (int posa = 0; posa < lena; posa++)
        {
            int idxa = d_blkcolidxA[abase+posa]; 
            int posb = binary_search_right_boundary_kernel(&d_blkrowidxB[bbase+posb_real], idxa, lenb-posb_real);
            if (posb < 0) continue;
            if (posb > lenb-posb_real) break;
            int idxb = d_blkrowidxB[bbase+posb_real+posb];

            if (idxa == idxb)
            {
                posb_real = posb_real + posb;
                if (lane_id < BLOCK_SIZE)
                {
                    s_blkmaskB_local[lane_id] = d_blkmaskB[(bbase+posb_real) * BLOCK_SIZE + lane_id];
                }

                const int nnzastart = d_nnzb_A[(abase+posa)];
                int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;

                for (int i = lane_id; i < nnztotala; i+= WARP_SIZE)
                {
                    unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+i];
                    unsigned int maskb = s_blkmaskB_local[rowcolidx & 0xf];
                    atomicOr(&s_maskc_local[rowcolidx >> 4], maskb);
                }
            }
        }
    }



/*

*/

/*
    int posa = posa_real;
    int posb = posb_real;
    int idxa = 0;
    int idxb = 0;
    int posa_updated = 1;
    int posb_updated = 1;
//int matched = 0;
    //while(posa < lena && posb < lenb)
    while(posa < lena && posb < lenb)
    {
        //idxa = posa_updated ? d_blkcolidxA[abase+posa] : idxa; //a[posa] : idxa;
        idxa = posa_updated ? d_blkcolidxA[abase+posa] : idxa; //a[posa] : idxa;
        //idxa = posa_updated ? (lena < 128 ? s_blkcolidxA[posa] : d_blkcolidxA[abase+posa]) : idxa; //a[posa] : idxa;

        //idxb = posb_updated ? d_blkrowidxB[bbase+posb] : idxb; //b[posb] : idxb;
        idxb = posb_updated ? d_blkrowidxB[bbase+posb] : idxb; //b[posb] : idxb;

        if (idxa == idxb)
        {
//matched++;
            if (lane_id < BLOCK_SIZE)
            {
                s_blkmaskB_local[lane_id] = d_blkmaskB[(bbase+posb) * BLOCK_SIZE + lane_id];
            }

            const int nnzastart = d_nnzb_A[(abase+posa)];
            int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;

            for (int i = lane_id; i < nnztotala; i+= WARP_SIZE)
            {
                unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+i];
                unsigned int maskb = s_blkmaskB_local[rowcolidx & 0xf];
                atomicOr(&s_maskc_local[rowcolidx >> 4], maskb);
            }

            // do spgemm of this pair
            //printf ("do spgemm of this pair, idxa = %i, idxb = %i\n", idxa, idxb);
            posa++;
            posa_updated = 1;
            posb++;
            posb_updated = 1;
        }
        else
        {
            // the smaller index goes forward
            //printf ("the smaller index goes forward (before), posa = %i, posb = %i\n", posa, posb);
            //posa = idxa < idxb ? posa + 1 : posa;
            posa_updated = idxa < idxb ? 1 : 0;
            posa += posa_updated;
            //posb = idxa > idxb ? posb + 1 : posb;
            posb_updated = idxa > idxb ? 1 : 0;
            posb += posb_updated;
            //printf ("the smaller index goes forward (after),  posa = %i, posb = %i\n", posa, posb);
        }
    }
*/

//    if (!lane_id) printf("%i,%i,%i,%i\n", global_warp_id, lena, lenb, matched);

maskc = lane_id < BLOCK_SIZE ? s_maskc_local[lane_id] : 0;
int nnzcnt = __popc(maskc); //lane_id < BLOCK_SIZE ? __popc(maskc) : 0;

//if (global_warp_id == 194909)
//printf("lane_id = %i, maskc = %i, nnzcnt = %i\n", lane_id, maskc, nnzcnt);

    int nnzcnt_sum = sum_32_shfl(nnzcnt);

if (nnzcnt_sum == 0)
{
    if (lane_id < BLOCK_SIZE) 
    {
        d_blkcsr_Ptr_C[global_warp_id * BLOCK_SIZE + lane_id] = 0;
        d_blkmaskC[global_warp_id * BLOCK_SIZE + lane_id] = 0;
    }
    if (!lane_id)
        d_nnzb_C[global_warp_id] = 0;
}
else
{
    int nnzcnt_scan = scan_32_shfl(nnzcnt, lane_id);
    if (lane_id < BLOCK_SIZE) 
        d_blkcsr_Ptr_C[global_warp_id * BLOCK_SIZE + lane_id] = nnzcnt_scan - nnzcnt;

    nnzcnt = __shfl_sync(0xffffffff, nnzcnt_scan, BLOCK_SIZE);
    //*s_blknnzC = nnzcnt;

    //int dnscnt = 0;
    //if (lane_id < BLOCK_SIZE)
    //{
    //    for (int i = 0; i < BLOCK_SIZE; i++)
    //        dnscnt = s_dnsidx_local[i * BLOCK_SIZE + lane_id] == 1 ? dnscnt + 1 : dnscnt;
    //}
    //__syncthreads();
    //dnscnt = sum_32_shfl(dnscnt);
    //__syncthreads();

    if (lane_id < BLOCK_SIZE && nnzcnt)
        d_blkmaskC[global_warp_id * BLOCK_SIZE + lane_id] = s_maskc_local[lane_id];

    if (!lane_id)
    {
        d_nnzb_C[global_warp_id] = nnzcnt;

        //printf("%i,%i\n", global_warp_id, nnzcnt);
/*
        if (nnzcnt <= 32)// && nnzcnt > 0)
        {
            int pos = atomicAdd(d_blksmem_cnt, 1);
            d_blkid_smem[pos] = global_warp_id;
        }
        else if (nnzcnt > 32)
        {
            int pos = atomicAdd(d_blkdns_cnt, 1);
            d_blkid_dns[pos] = global_warp_id;
        }
*/
    }
}
}


template <int SMEM_SIZE>
__global__
void stir_spgemm_step3_cuda_kernel_smem
                             (int *d_blkrowptrA,
                            int *d_blkcolidxA,
                            int *d_nnzb_A,
                            MAT_VAL_TYPE *d_blkcsr_Val_A,
                            unsigned char *d_blkcsr_Col_A,
                            unsigned char *d_blkcsr_Ptr_A,
                            unsigned short *d_blkmaskA,
                            int blkmA, int blknA, int numblkA, int nnzA, 
                            int *d_blkcolptrB,
                            int *d_blkrowidxB, 
                            int *d_nnzb_B,
                            MAT_VAL_TYPE *d_blkcsr_Val_B,
                            unsigned char *d_blkcsr_Col_B,
                            unsigned char *d_blkcsr_Ptr_B,
                            unsigned short *d_blkmaskB,
                            int blkmB, int blknB , int numblkB, int nnzB, 
                            int *d_blkrowidxC,
                            int *d_blkcolidxC,
                            unsigned char *d_blkcsr_Ptr_C,
                            int *d_nnzb_C,
                            unsigned short *d_blkmaskC,
                            int *d_blksmem_cnt,
                            int *d_blkdns_cnt,
                            int *d_blkid_smem,
                            int *d_blkid_dns,
                            int numblkC)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp_id = global_id / WARP_SIZE;

    if (global_warp_id >= numblkC) return;
    //if (global_warp_id < numblkC)

    __shared__ unsigned int s_maskc[WARP_PER_BLOCK * BLOCK_SIZE];
    //__shared__ unsigned short s_blkmaskA[WARP_PER_BLOCK * BLOCK_SIZE];
    __shared__ unsigned short s_blkmaskB[WARP_PER_BLOCK * BLOCK_SIZE];
    __shared__ int s_blkcolidxA[WARP_PER_BLOCK * SMEM_SIZE];
    __shared__ int s_blkrowidxB[WARP_PER_BLOCK * SMEM_SIZE];

    //const int local_warp_id = threadIdx.x / WARP_SIZE;
    //__shared__ char s_dnsidx[WARP_PER_BLOCK * BLOCK_SIZE * BLOCK_SIZE];
    //char *s_dnsidx_local = &s_dnsidx[local_warp_id * BLOCK_SIZE * BLOCK_SIZE];

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    //const int sublane_id = lane_id % (WARP_SIZE / BLOCK_SIZE);
    //const int subwarp_id = lane_id / (WARP_SIZE / BLOCK_SIZE);
    unsigned int maskc = 0;
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    
    unsigned int *s_maskc_local = &s_maskc[local_warp_id * BLOCK_SIZE];
    //unsigned short *s_blkmaskA_local = &s_blkmaskA[local_warp_id * BLOCK_SIZE];
    unsigned short *s_blkmaskB_local = &s_blkmaskB[local_warp_id * BLOCK_SIZE];
    int *s_blkcolidxA_local = &s_blkcolidxA[local_warp_id * SMEM_SIZE];
    int *s_blkrowidxB_local = &s_blkrowidxB[local_warp_id * SMEM_SIZE];

    //for (int i = 0; i < BLOCK_SIZE; i++)
    //    if (lane_id < BLOCK_SIZE)
    //        s_dnsidx_local[i * BLOCK_SIZE + lane_id] = 0;
    //__syncthreads();
    if (lane_id < BLOCK_SIZE)
        s_maskc_local[lane_id] = 0;
    //__syncthreads();

    const int blki = d_blkrowidxC[global_warp_id];
    const int blkj = d_blkcolidxC[global_warp_id];

    const int abase = d_blkrowptrA[blki];
    //const int astop = d_blkrowptrA[blki+1];
    //const int astart = d_blkcolidxA[abase];
    //const int aend = d_blkcolidxA[astop-1];
    const int lena = d_blkrowptrA[blki+1] - abase;

    const int bbase = d_blkcolptrB[blkj];
    //const int bstop = d_blkcolptrB[blkj+1];
    //const int bstart = d_blkrowidxB[bbase];
    //const int bend = d_blkrowidxB[bstop-1];
    const int lenb = d_blkcolptrB[blkj+1] - bbase;

//    if (!lane_id) printf("%i,%i,%i\n", global_warp_id, lena, lenb);
//return;

    int loadlena = lena > SMEM_SIZE ? SMEM_SIZE : lena;
    for (int i = lane_id; i < loadlena; i += WARP_SIZE)
    {
        s_blkcolidxA_local[i] = d_blkcolidxA[abase + i];
    }

    int loadlenb = lenb > SMEM_SIZE ? SMEM_SIZE : lenb;
    for (int i = lane_id; i < loadlenb; i += WARP_SIZE)
    {
        s_blkrowidxB_local[i] = d_blkrowidxB[bbase + i];
    }
    //__syncthreads();

    const int astart = s_blkcolidxA_local[0];
    const int bstart = s_blkrowidxB_local[0];

    int posa_real = 0;
    int posb_real = 0;

    if (bstart > astart)
    {
        //posa_real = binary_search_right_boundary_kernel((lena <= 128 ? s_blkcolidxA : &d_blkcolidxA[abase]), bstart, lena);
        //posa_real = binary_search_right_boundary_kernel(&d_blkcolidxA[abase], bstart, lena);
        posa_real = binary_search_right_boundary_kernel((lena <= SMEM_SIZE ? s_blkcolidxA_local : &d_blkcolidxA[abase]), bstart, lena);
    }
    else if (bstart < astart)
    {
        //posb_real = binary_search_right_boundary_kernel(&d_blkrowidxB[bbase], astart, lenb);
        posb_real = binary_search_right_boundary_kernel((lenb <= SMEM_SIZE ? s_blkrowidxB_local : &d_blkrowidxB[bbase]), astart, lenb);
    }
/*
    for (int posa = 0; posa < lena; posa++)
    {
        int idxa = d_blkcolidxA[abase+posa]; 
        int posb = binary_search_right_boundary_kernel(&d_blkrowidxB[bbase+posb_real], idxa, lenb-posb_real);
        if (posb < 0) continue;
        if (posb > lenb-posb_real) break;
        int idxb = d_blkrowidxB[bbase+posb_real+posb];

        if (idxa == idxb)
        {
            posb_real = posb_real + posb;
            if (lane_id < BLOCK_SIZE)
            {
                s_blkmaskB_local[lane_id] = d_blkmaskB[(bbase+posb_real) * BLOCK_SIZE + lane_id];
            }

            const int nnzastart = d_nnzb_A[(abase+posa)];
            int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;

            for (int i = lane_id; i < nnztotala; i+= WARP_SIZE)
            {
                unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+i];
                unsigned int maskb = s_blkmaskB_local[rowcolidx & 0xf];
                atomicOr(&s_maskc_local[rowcolidx >> 4], maskb);
            }
        }
    }
*/


    int posa = posa_real;
    int posb = posb_real;
    int idxa = 0;
    int idxb = 0;
    int posa_updated = 1;
    int posb_updated = 1;

    //while(posa < lena && posb < lenb)
    while(posa < lena && posb < lenb)
    {
        //idxa = posa_updated ? d_blkcolidxA[abase+posa] : idxa; //a[posa] : idxa;
        idxa = posa_updated ? (lena < SMEM_SIZE ? s_blkcolidxA_local[posa] : d_blkcolidxA[abase+posa]) : idxa; //a[posa] : idxa;
        //idxa = posa_updated ? (lena < 128 ? s_blkcolidxA[posa] : d_blkcolidxA[abase+posa]) : idxa; //a[posa] : idxa;

        //idxb = posb_updated ? d_blkrowidxB[bbase+posb] : idxb; //b[posb] : idxb;
        idxb = posb_updated ? (lenb < SMEM_SIZE ? s_blkrowidxB_local[posb] : d_blkrowidxB[bbase+posb]) : idxb; //b[posb] : idxb;

        if (idxa == idxb)
        {
            if (lane_id < BLOCK_SIZE)
            {
                s_blkmaskB_local[lane_id] = d_blkmaskB[(bbase+posb) * BLOCK_SIZE + lane_id];
            }

            const int nnzastart = d_nnzb_A[(abase+posa)];
            int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;

            for (int i = lane_id; i < nnztotala; i+= WARP_SIZE)
            {
                unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+i];
                unsigned int maskb = s_blkmaskB_local[rowcolidx & 0xf];
                atomicOr(&s_maskc_local[rowcolidx >> 4], maskb);
            }

            // do spgemm of this pair
            //printf ("do spgemm of this pair, idxa = %i, idxb = %i\n", idxa, idxb);
            posa++;
            posa_updated = 1;
            posb++;
            posb_updated = 1;
        }
        else
        {
            // the smaller index goes forward
            //printf ("the smaller index goes forward (before), posa = %i, posb = %i\n", posa, posb);
            //posa = idxa < idxb ? posa + 1 : posa;
            posa_updated = idxa < idxb ? 1 : 0;
            posa += posa_updated;
            //posb = idxa > idxb ? posb + 1 : posb;
            posb_updated = idxa > idxb ? 1 : 0;
            posb += posb_updated;
            //printf ("the smaller index goes forward (after),  posa = %i, posb = %i\n", posa, posb);
        }
    }


maskc = lane_id < BLOCK_SIZE ? s_maskc_local[lane_id] : 0;
int nnzcnt = __popc(maskc); //lane_id < BLOCK_SIZE ? __popc(maskc) : 0;

//if (global_warp_id == 194909)
//printf("lane_id = %i, maskc = %i, nnzcnt = %i\n", lane_id, maskc, nnzcnt);

    //nnzcnt = sum_32_shfl(nnzcnt);
    
    int nnzcnt_scan = scan_32_shfl(nnzcnt, lane_id);
    if (lane_id < BLOCK_SIZE) 
        d_blkcsr_Ptr_C[global_warp_id * BLOCK_SIZE + lane_id] = nnzcnt_scan - nnzcnt;

    nnzcnt = __shfl_sync(0xffffffff, nnzcnt_scan, BLOCK_SIZE);
    //*s_blknnzC = nnzcnt;

    //int dnscnt = 0;
    //if (lane_id < BLOCK_SIZE)
    //{
    //    for (int i = 0; i < BLOCK_SIZE; i++)
    //        dnscnt = s_dnsidx_local[i * BLOCK_SIZE + lane_id] == 1 ? dnscnt + 1 : dnscnt;
    //}
    //__syncthreads();
    //dnscnt = sum_32_shfl(dnscnt);
    //__syncthreads();

    if (lane_id < BLOCK_SIZE && nnzcnt)
        d_blkmaskC[global_warp_id * BLOCK_SIZE + lane_id] = s_maskc_local[lane_id];

    if (!lane_id)
    {
        d_nnzb_C[global_warp_id] = nnzcnt;

        //printf("%i,%i\n", global_warp_id, nnzcnt);
/*
        if (nnzcnt <= 32)// && nnzcnt > 0)
        {
            int pos = atomicAdd(d_blksmem_cnt, 1);
            d_blkid_smem[pos] = global_warp_id;
        }
        else if (nnzcnt > 32)
        {
            int pos = atomicAdd(d_blkdns_cnt, 1);
            d_blkid_dns[pos] = global_warp_id;
        }
*/
    }

}

__global__
void stir_spgemm_step3_cuda_kernel_reg
                             (int *d_blkrowptrA,
                            int *d_blkcolidxA,
                            int *d_nnzb_A,
                            MAT_VAL_TYPE *d_blkcsr_Val_A,
                            unsigned char *d_blkcsr_Col_A,
                            unsigned char *d_blkcsr_Ptr_A,
                            unsigned short *d_blkmaskA,
                            int blkmA, int blknA, int numblkA, int nnzA, 
                            int *d_blkcolptrB,
                            int *d_blkrowidxB, 
                            int *d_nnzb_B,
                            MAT_VAL_TYPE *d_blkcsr_Val_B,
                            unsigned char *d_blkcsr_Col_B,
                            unsigned char *d_blkcsr_Ptr_B,
                            unsigned short *d_blkmaskB,
                            int blkmB, int blknB , int numblkB, int nnzB, 
                            int *d_blkrowidxC,
                            int *d_blkcolidxC,
                            unsigned char *d_blkcsr_Ptr_C,
                            int *d_nnzb_C,
                            unsigned short *d_blkmaskC,
                            int *d_blksmem_cnt,
                            int *d_blkdns_cnt,
                            int *d_blkid_smem,
                            int *d_blkid_dns,
                            int numblkC)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp_id = global_id / WARP_SIZE;

    if (global_warp_id >= numblkC) return;
    //if (global_warp_id < numblkC)

    __shared__ unsigned int s_maskc[WARP_PER_BLOCK * BLOCK_SIZE];
    //__shared__ unsigned short s_blkmaskA[WARP_PER_BLOCK * BLOCK_SIZE];
    __shared__ unsigned short s_blkmaskB[WARP_PER_BLOCK * BLOCK_SIZE];

    //const int local_warp_id = threadIdx.x / WARP_SIZE;
    //__shared__ char s_dnsidx[WARP_PER_BLOCK * BLOCK_SIZE * BLOCK_SIZE];
    //char *s_dnsidx_local = &s_dnsidx[local_warp_id * BLOCK_SIZE * BLOCK_SIZE];

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    //const int sublane_id = lane_id % (WARP_SIZE / BLOCK_SIZE);
    //const int subwarp_id = lane_id / (WARP_SIZE / BLOCK_SIZE);
    unsigned int maskc = 0;
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    
    unsigned int *s_maskc_local = &s_maskc[local_warp_id * BLOCK_SIZE];
    //unsigned short *s_blkmaskA_local = &s_blkmaskA[local_warp_id * BLOCK_SIZE];
    unsigned short *s_blkmaskB_local = &s_blkmaskB[local_warp_id * BLOCK_SIZE];

    //for (int i = 0; i < BLOCK_SIZE; i++)
    //    if (lane_id < BLOCK_SIZE)
    //        s_dnsidx_local[i * BLOCK_SIZE + lane_id] = 0;
    //__syncthreads();
    if (lane_id < BLOCK_SIZE)
        s_maskc_local[lane_id] = 0;
    //__syncthreads();

    const int blki = d_blkrowidxC[global_warp_id];
    const int blkj = d_blkcolidxC[global_warp_id];

    const int abase = d_blkrowptrA[blki];
    //const int astop = d_blkrowptrA[blki+1];
    const int astart = d_blkcolidxA[abase];
    //const int aend = d_blkcolidxA[astop-1];
    const int lena = d_blkrowptrA[blki+1] - abase;

    const int bbase = d_blkcolptrB[blkj];
    //const int bstop = d_blkcolptrB[blkj+1];
    const int bstart = d_blkrowidxB[bbase];
    //const int bend = d_blkrowidxB[bstop-1];
    const int lenb = d_blkcolptrB[blkj+1] - bbase;

//    if (!lane_id) printf("%i,%i,%i\n", global_warp_id, lena, lenb);
//return;

    int posa_real = 0;
    int posb_real = 0;

    if (bstart > astart)
    {
        //posa_real = binary_search_right_boundary_kernel((lena <= 128 ? s_blkcolidxA : &d_blkcolidxA[abase]), bstart, lena);
        posa_real = binary_search_right_boundary_kernel(&d_blkcolidxA[abase], bstart, lena);
        //posa_real = binary_search_right_boundary_kernel((lena <= SMEM_SIZE ? s_blkcolidxA_local : &d_blkcolidxA[abase]), bstart, lena);
    }
    else if (bstart < astart)
    {
        posb_real = binary_search_right_boundary_kernel(&d_blkrowidxB[bbase], astart, lenb);
        //posb_real = binary_search_right_boundary_kernel((lenb <= SMEM_SIZE ? s_blkrowidxB_local : &d_blkrowidxB[bbase]), astart, lenb);
    }
/*
    for (int posa = 0; posa < lena; posa++)
    {
        int idxa = d_blkcolidxA[abase+posa]; 
        int posb = binary_search_right_boundary_kernel(&d_blkrowidxB[bbase+posb_real], idxa, lenb-posb_real);
        if (posb < 0) continue;
        if (posb > lenb-posb_real) break;
        int idxb = d_blkrowidxB[bbase+posb_real+posb];

        if (idxa == idxb)
        {
            posb_real = posb_real + posb;
            if (lane_id < BLOCK_SIZE)
            {
                s_blkmaskB_local[lane_id] = d_blkmaskB[(bbase+posb_real) * BLOCK_SIZE + lane_id];
            }

            const int nnzastart = d_nnzb_A[(abase+posa)];
            int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;

            for (int i = lane_id; i < nnztotala; i+= WARP_SIZE)
            {
                unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+i];
                unsigned int maskb = s_blkmaskB_local[rowcolidx & 0xf];
                atomicOr(&s_maskc_local[rowcolidx >> 4], maskb);
            }
        }
    }
*/

    int posa = posa_real;
    int posb = posb_real;
    int idxa = 0;
    int idxb = 0;
    int posa_updated = 1;
    int posb_updated = 1;

    int r_blkcolidxA = posa+lane_id < lena ? d_blkcolidxA[abase+posa+lane_id] : 0;
    int r_blkrowidxB = posb+lane_id < lenb ? d_blkrowidxB[bbase+posb+lane_id] : 0;
    int cnta = 0;
    int cntb = 0;

    //while(posa < lena && posb < lenb)
    while(posa < lena && posb < lenb)
    {
        r_blkcolidxA = cnta == WARP_SIZE ? (posa+lane_id < lena ? d_blkcolidxA[abase+posa+lane_id] : 0) : r_blkcolidxA;
        cnta = cnta == WARP_SIZE ? 0 : cnta;

        r_blkrowidxB = cntb == WARP_SIZE ? (posb+lane_id < lenb ? d_blkrowidxB[bbase+posb+lane_id] : 0) : r_blkrowidxB;
        cntb = cntb == WARP_SIZE ? 0 : cntb;

        //idxa = posa_updated ? d_blkcolidxA[abase+posa] : idxa; //a[posa] : idxa;
        idxa = posa_updated ? __shfl_sync(0xffffffff, r_blkcolidxA, cnta) : idxa; //a[posa] : idxa;

        //idxb = posb_updated ? d_blkrowidxB[bbase+posb] : idxb; //b[posb] : idxb;
        idxb = posb_updated ? __shfl_sync(0xffffffff, r_blkrowidxB, cntb) : idxb; //b[posb] : idxb;

        if (idxa == idxb)
        {
            if (lane_id < BLOCK_SIZE)
            {
                s_blkmaskB_local[lane_id] = d_blkmaskB[(bbase+posb) * BLOCK_SIZE + lane_id];
            }

            const int nnzastart = d_nnzb_A[(abase+posa)];
            int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;

            for (int i = lane_id; i < nnztotala; i+= WARP_SIZE)
            {
                unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+i];
                unsigned int maskb = s_blkmaskB_local[rowcolidx & 0xf];
                atomicOr(&s_maskc_local[rowcolidx >> 4], maskb);
            }

            // do spgemm of this pair
            //printf ("do spgemm of this pair, idxa = %i, idxb = %i\n", idxa, idxb);
            posa++;
            cnta++;
            posa_updated = 1;

            posb++;
            cntb++;
            posb_updated = 1;
        }
        else
        {
            // the smaller index goes forward
            //printf ("the smaller index goes forward (before), posa = %i, posb = %i\n", posa, posb);
            //posa = idxa < idxb ? posa + 1 : posa;
            posa_updated = idxa < idxb ? 1 : 0;
            posa += posa_updated;
            cnta += posa_updated;

            //posb = idxa > idxb ? posb + 1 : posb;
            posb_updated = idxa > idxb ? 1 : 0;
            posb += posb_updated;
            cntb += posb_updated;
            //printf ("the smaller index goes forward (after),  posa = %i, posb = %i\n", posa, posb);
        }
    }


maskc = lane_id < BLOCK_SIZE ? s_maskc_local[lane_id] : 0;
int nnzcnt = __popc(maskc); //lane_id < BLOCK_SIZE ? __popc(maskc) : 0;

//if (global_warp_id == 194909)
//printf("lane_id = %i, maskc = %i, nnzcnt = %i\n", lane_id, maskc, nnzcnt);

    //nnzcnt = sum_32_shfl(nnzcnt);
    
    int nnzcnt_scan = scan_32_shfl(nnzcnt, lane_id);
    if (lane_id < BLOCK_SIZE) 
        d_blkcsr_Ptr_C[global_warp_id * BLOCK_SIZE + lane_id] = nnzcnt_scan - nnzcnt;

    nnzcnt = __shfl_sync(0xffffffff, nnzcnt_scan, BLOCK_SIZE);
    //*s_blknnzC = nnzcnt;

    //int dnscnt = 0;
    //if (lane_id < BLOCK_SIZE)
    //{
    //    for (int i = 0; i < BLOCK_SIZE; i++)
    //        dnscnt = s_dnsidx_local[i * BLOCK_SIZE + lane_id] == 1 ? dnscnt + 1 : dnscnt;
    //}
    //__syncthreads();
    //dnscnt = sum_32_shfl(dnscnt);
    //__syncthreads();

    if (lane_id < BLOCK_SIZE && nnzcnt)
        d_blkmaskC[global_warp_id * BLOCK_SIZE + lane_id] = s_maskc_local[lane_id];

    if (!lane_id)
    {
        d_nnzb_C[global_warp_id] = nnzcnt;

        //printf("%i,%i\n", global_warp_id, nnzcnt);
/*
        if (nnzcnt <= 32)// && nnzcnt > 0)
        {
            int pos = atomicAdd(d_blksmem_cnt, 1);
            d_blkid_smem[pos] = global_warp_id;
        }
        else if (nnzcnt > 32)
        {
            int pos = atomicAdd(d_blkdns_cnt, 1);
            d_blkid_dns[pos] = global_warp_id;
        }
*/
    }

}


void step3_cuda (int *blkrowptrA, int *blkcolidxA, int blkmA, int blknA, int numblkA, int nnzA, int *nnzb_A ,int mA,
            MAT_VAL_TYPE *blkcsr_Val_A , unsigned char  *blkcsr_Col_A , unsigned char *blkcsr_Ptr_A , unsigned short *blkmaskA,
            int *blkcolptrB, int *blkrowidxB, int blkmB, int blknB , int numblkB, int nnzB, int *nnzb_B ,
            MAT_VAL_TYPE *blkcsr_Val_B , unsigned char  *blkcsr_Col_B , unsigned char *blkcsr_Ptr_B ,unsigned short *blkmaskB,
            int *blkrowptrC, int *blkcolidxC,int *d_blkrowidxC, int *d_blkcolidxC, unsigned char *d_blkcsr_Ptr_C, unsigned short *d_blkmaskC,
            int *nnzb_C_golden,  int numblkC, int *nnzC, int *d_blksmem_cnt, int *d_blkdns_cnt, int *d_blkid_smem, int *d_blkid_dns)
{
    int *d_blkrowptrA;
    int *d_blkcolidxA; 
    int *d_nnzb_A; 
    MAT_VAL_TYPE *d_blkcsr_Val_A;
    unsigned char *d_blkcsr_Col_A;
    unsigned char *d_blkcsr_Ptr_A;
    unsigned short *d_blkmaskA;

    cudaMalloc((void **)&d_blkrowptrA, (blkmA+1) * sizeof(int));
    cudaMalloc((void **)&d_blkcolidxA, numblkA * sizeof(int));
    cudaMalloc((void **)&d_nnzb_A, (numblkA+1) * sizeof(int));
    cudaMalloc((void **)&d_blkcsr_Val_A, nnzA * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_blkcsr_Col_A, nnzA * sizeof(unsigned char));
    cudaMalloc((void **)&d_blkcsr_Ptr_A, numblkA * BLOCK_SIZE  * sizeof(unsigned char));
    cudaMalloc((void **)&d_blkmaskA, numblkA * BLOCK_SIZE  * sizeof(unsigned short));

    cudaMemcpy(d_blkrowptrA,     blkrowptrA,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcolidxA,     blkcolidxA,     numblkA * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnzb_A,     nnzb_A,     (numblkA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Val_A,     blkcsr_Val_A,     nnzA * sizeof(MAT_VAL_TYPE),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Col_A,     blkcsr_Col_A,     nnzA * sizeof(unsigned char),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Ptr_A,     blkcsr_Ptr_A,     numblkA * BLOCK_SIZE * sizeof(unsigned char),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkmaskA,     blkmaskA,     numblkA * BLOCK_SIZE * sizeof(unsigned short),              cudaMemcpyHostToDevice);

    int *d_blkcolptrB;
    int *d_blkrowidxB; 
    int *d_nnzb_B; 
    MAT_VAL_TYPE *d_blkcsr_Val_B;
    unsigned char *d_blkcsr_Col_B;
    unsigned char *d_blkcsr_Ptr_B;
    unsigned short *d_blkmaskB;

    cudaMalloc((void **)&d_blkcolptrB, (blknB+1) * sizeof(int));
    cudaMalloc((void **)&d_blkrowidxB, numblkB * sizeof(int));
    cudaMalloc((void **)&d_nnzb_B, (numblkB+1) * sizeof(int));
    cudaMalloc((void **)&d_blkcsr_Val_B, nnzB * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_blkcsr_Col_B, nnzB * sizeof(unsigned char));
    cudaMalloc((void **)&d_blkcsr_Ptr_B, numblkB * BLOCK_SIZE  * sizeof(unsigned char));
    cudaMalloc((void **)&d_blkmaskB, numblkB * BLOCK_SIZE  * sizeof(unsigned short));

    cudaMemcpy(d_blkcolptrB,     blkcolptrB,     (blknB+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkrowidxB,     blkrowidxB,     numblkB * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnzb_B,     nnzb_B,     (numblkB+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Val_B,     blkcsr_Val_B,     nnzB * sizeof(MAT_VAL_TYPE),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Col_B,     blkcsr_Col_B,     nnzB * sizeof(unsigned char),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Ptr_B,     blkcsr_Ptr_B,     numblkB * BLOCK_SIZE * sizeof(unsigned char),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkmaskB,     blkmaskB,     numblkB * BLOCK_SIZE * sizeof(unsigned short),              cudaMemcpyHostToDevice);

    //int *d_blkrowptrC;
    //int *d_blkrowidxC;
    //int *d_blkcolidxC;
    int *d_nnzb_C;

    //cudaMalloc((void **)&d_blkrowptrC, (blkmA+1) * sizeof(int));
    //cudaMalloc((void **)&d_blkrowidxC, numblkC * sizeof(int));
    //cudaMalloc((void **)&d_blkcolidxC, numblkC * sizeof(int));
    cudaMalloc((void **)&d_nnzb_C, (numblkC+1) * sizeof(int));



    //cudaMemcpy(d_blkrowptrC,     blkrowptrC,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemcpy(d_blkrowidxC,     blkrowidxC,     numblkC * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemcpy(d_blkcolidxC,     blkcolidxC,     numblkC * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemset(d_nnzb_C, 0, (numblkC+1) * sizeof(int));


    struct timeval t1, t2;

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

    int num_threads = WARP_SIZE * WARP_PER_BLOCK;
    int num_blocks = ceil((double)numblkC/(double)WARP_PER_BLOCK);
    stir_spgemm_step3_cuda_kernel<128><<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A, d_blkmaskA,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B, d_blkmaskB,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, d_nnzb_C, d_blkmaskC, 
                              d_blksmem_cnt, d_blkdns_cnt, d_blkid_smem, d_blkid_dns, numblkC);

exclusive_scan_device_cuda(d_nnzb_C, numblkC+1);
int nbc = 0;
cudaMemcpy(&nbc,     &d_nnzb_C[numblkC],     sizeof(int), cudaMemcpyDeviceToHost);
//*numblkC = nbc;

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

int blksmem_cnt = 0;
int blkdns_cnt = 0;
cudaMemcpy(&blksmem_cnt,     d_blksmem_cnt,     sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(&blkdns_cnt,     d_blkdns_cnt,     sizeof(int), cudaMemcpyDeviceToHost);

printf("blksmem_cnt = %i, blkdns_cnt = %i\n", blksmem_cnt, blkdns_cnt);

    double time_kernel = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("CUDA step3 kernel = %.2f ms, nnzC = %i\n", time_kernel, nbc);

    int *h_nnzb_C = (int *)malloc((numblkC+1)*sizeof(int));
    cudaMemcpy(h_nnzb_C,     d_nnzb_C,     (numblkC+1) * sizeof(int), cudaMemcpyDeviceToHost);

    int errcnt = 0;
    for (int i = 0; i < (numblkC+1); i++)
    //for (int i = 194905; i < (numblkC+1); i++)
        if (h_nnzb_C[i] != nnzb_C_golden[i])
            {//printf("[%i] %i, %i\n", i, h_nnzb_C[i], nnzb_C_golden[i]); 
errcnt++;}
    printf("step 3, h_nnzb_C, errcnt = %i\n", errcnt);

    //printf("move %i\n", 0x1 << 2);  //0001 -> 0010

    cudaFree(d_blkrowptrA);
    cudaFree(d_blkcolidxA); 
    cudaFree(d_nnzb_A); 
    cudaFree(d_blkcsr_Val_A);
    cudaFree(d_blkcsr_Col_A);
    cudaFree(d_blkcsr_Ptr_A);

    cudaFree(d_blkcolptrB);
    cudaFree(d_blkrowidxB); 
    cudaFree(d_nnzb_B); 
    cudaFree(d_blkcsr_Val_B);
    cudaFree(d_blkcsr_Col_B);
    cudaFree(d_blkcsr_Ptr_B);
    cudaFree(d_blkmaskB);

    //cudaFree(d_blkrowptrC);
    //cudaFree(d_blkrowidxC);
    //cudaFree(d_blkcolidxC);
    cudaFree(d_nnzb_C);
    free(h_nnzb_C);
}


template <int SMEM_SIZE>
__global__
void stir_spgemm_step4_cuda_kernel_smem
                             (int *d_blkrowptrA,
                            const int* __restrict__ d_blkcolidxA,
                            int *d_nnzb_A,
                            MAT_VAL_TYPE *d_blkcsr_Val_A,
                            unsigned char *d_blkcsr_Col_A,
                            unsigned char *d_blkcsr_Ptr_A,
                            int blkmA, int blknA, int numblkA, int nnzA, 
                            int *d_blkcolptrB,
                            const int* __restrict__ d_blkrowidxB, 
                            int *d_nnzb_B,
                            MAT_VAL_TYPE *d_blkcsr_Val_B,
                            unsigned char *d_blkcsr_Col_B,
                            unsigned char *d_blkcsr_Ptr_B,
                            int blkmB, int blknB , int numblkB, int nnzB, 
                            int *d_blkrowidxC,
                            int *d_blkcolidxC,
                            unsigned char *d_blkcsr_Ptr_C,
                            unsigned char *d_blkcsr_Col_C,
                            MAT_VAL_TYPE *d_blkcsr_Val_C,
                            int *d_nnzb_C,
                            unsigned short *d_blkmaskC,
                            int numblkC,
                            int *d_blkid)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
     int global_warp_id = global_id / WARP_SIZE;

    if (global_warp_id >= numblkC) return;
    global_warp_id = d_blkid[global_warp_id];

    const int nnzcstart = d_nnzb_C[global_warp_id];
    const int blknnzctotal = d_nnzb_C[global_warp_id+1] - nnzcstart;
    if (!blknnzctotal) return;

    const int local_warp_id = threadIdx.x / WARP_SIZE;
    __shared__ unsigned char s_blkcsr_Idx_C[WARP_PER_BLOCK * SMEM_SIZE];
    __shared__ MAT_VAL_TYPE s_blkcsr_Val_C[WARP_PER_BLOCK * SMEM_SIZE];
    __shared__ unsigned char s_blkcsr_Ptr_C[WARP_PER_BLOCK * BLOCK_SIZE];
    unsigned char *s_blkcsr_Idx_C_local = &s_blkcsr_Idx_C[local_warp_id * SMEM_SIZE];
    MAT_VAL_TYPE *s_blkcsr_Val_C_local = &s_blkcsr_Val_C[local_warp_id * SMEM_SIZE];
    unsigned char *s_blkcsr_Ptr_C_local = &s_blkcsr_Ptr_C[local_warp_id * BLOCK_SIZE];


    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int sublane_id = lane_id % (WARP_SIZE / BLOCK_SIZE);
    const int subwarp_id = lane_id / (WARP_SIZE / BLOCK_SIZE);

    //__shared__ unsigned int s_maskc[WARP_PER_BLOCK * BLOCK_SIZE];
    //unsigned int *s_maskc_local = &s_maskc[local_warp_id * BLOCK_SIZE];

    for (int i = lane_id; i < SMEM_SIZE; i+=WARP_SIZE)
        s_blkcsr_Val_C_local[i] = 0;
    if (lane_id < BLOCK_SIZE)
        s_blkcsr_Ptr_C_local[lane_id] = d_blkcsr_Ptr_C[global_warp_id * BLOCK_SIZE + lane_id];
    //__syncthreads();

    unsigned int maskc = lane_id < BLOCK_SIZE ? d_blkmaskC[global_warp_id * BLOCK_SIZE + lane_id] : 0;//s_maskc_local[lane_id];
    unsigned char blknnzcstart = lane_id < BLOCK_SIZE ? s_blkcsr_Ptr_C_local[lane_id] : 0;

    // build s_blkcsr_Idx_C_local
    if (lane_id < BLOCK_SIZE)
    {
        int cnt = 0;
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            int idx = ((maskc >> BLOCK_SIZE - i - 1) & 0x1) == 1 ? i : -1;
            if (idx != -1)
            {
                s_blkcsr_Idx_C_local[blknnzcstart + cnt] = idx;
                cnt++;
            }
        }
    }

    const int blki = d_blkrowidxC[global_warp_id];
    const int blkj = d_blkcolidxC[global_warp_id];

    const int abase = d_blkrowptrA[blki];
    const int astop = d_blkrowptrA[blki+1];
    const int astart = d_blkcolidxA[abase];
    //const int aend = d_blkcolidxA[astop-1];
    const int lena = astop - abase;

    const int bbase = d_blkcolptrB[blkj];
    const int bstop = d_blkcolptrB[blkj+1];
    const int bstart = d_blkrowidxB[bbase];
    //const int bend = d_blkrowidxB[bstop-1];
    const int lenb = bstop - bbase;

    int posa_real = 0;
    int posb_real = 0;
    if (bstart > astart)
    {
        //posa_real = binary_search_right_boundary_kernel((lena <= 128 ? s_blkcolidxA : &d_blkcolidxA[abase]), bstart, lena);
        posa_real = binary_search_right_boundary_kernel(&d_blkcolidxA[abase], bstart, lena);
    }
    else if (bstart < astart)
    {
        posb_real = binary_search_right_boundary_kernel(&d_blkrowidxB[bbase], astart, lenb);
    }

    int posa = posa_real;
    int posb = posb_real;
    int idxa = 0;
    int idxb = 0;
    int posa_updated = 1;
    int posb_updated = 1;

    while(posa < lena && posb < lenb)
    {
        idxa = posa_updated ? d_blkcolidxA[abase+posa] : idxa; //a[posa] : idxa;
        //idxa = posa_updated ? (lena < 128 ? s_blkcolidxA[posa] : d_blkcolidxA[abase+posa]) : idxa; //a[posa] : idxa;
        idxb = posb_updated ? d_blkrowidxB[bbase+posb] : idxb; //b[posb] : idxb;

        if (idxa == idxb)
        {
             const int nnzastart = d_nnzb_A[(abase+posa)];
            int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;
            unsigned char *d_csrRowPtrB = &d_blkcsr_Ptr_B[(bbase+posb)*BLOCK_SIZE];
            const int nnzbstart = d_nnzb_B[(bbase+posb)];
            int nnztotalb = d_nnzb_B[(bbase+posb) + 1] - nnzbstart;

            for (int i = lane_id; i < nnztotala; i+= WARP_SIZE)
            {
                unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+i];
                int rowidxa = rowcolidx >> 4;
                int rowidxb = rowcolidx & 0xf;
                MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart+i];
                int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];

                const int startb = d_csrRowPtrB[rowidxb];
                const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : d_csrRowPtrB[rowidxb+1];
                for (int k = startb; k < stopb; k++)
                {
                    //unsigned char colidx_packet = d_csrColIdxB[k/2];
                    //unsigned char colidx = (k % 2 == 0) ? (colidx >> 4 & 0x0F) : (colidx & 0x0F);
                    unsigned char colidx = d_blkcsr_Col_B[nnzbstart+k];
                    MAT_VAL_TYPE valb = d_blkcsr_Val_B[nnzbstart+k];
                    //maskc = maskc | (0x1 << (BLOCK_SIZE - colidx - 1));
                    //atomicOr(&s_maskc_local[rowidxa], (unsigned int)(0x1 << (BLOCK_SIZE - colidx - 1)));
                    //s_dnsidx_local[subwarp_id * BLOCK_SIZE + colidx] = 1;
                    //s_blkcsr_Val_C_local[subwarp_id * BLOCK_SIZE + colidx] += val * valb;
                    //atomicAdd(&s_blkcsr_Val_C_local[rowidxa * BLOCK_SIZE + colidx], val * valb);

                    int cnt = 0;
                    //unsigned char colidx_exist = s_blkcsr_Idx_C_local[blknnzcstart];
                    while (colidx != s_blkcsr_Idx_C_local[blkoffseta + cnt])
                    {
                        cnt++;
                        //colidx_exist = s_blkcsr_Idx_C_local[blknnzcstart + cnt];
                    }
                    atomicAdd(&s_blkcsr_Val_C_local[blkoffseta + cnt], val * valb);
                    //atomicAdd(&s_blkcsr_Val_C_local[colidx * BLOCK_SIZE + rowidxa], val * valb);
                }
            }

            // do spgemm of this pair
            //printf ("do spgemm of this pair, idxa = %i, idxb = %i\n", idxa, idxb);
            posa++;
            posa_updated = 1;
            posb++;
            posb_updated = 1;
        }
        else
        {
            // the smaller index goes forward
            //printf ("the smaller index goes forward (before), posa = %i, posb = %i\n", posa, posb);
            //posa = idxa < idxb ? posa + 1 : posa;
            posa_updated = idxa < idxb ? 1 : 0;
            posa += posa_updated;
            //posb = idxa > idxb ? posb + 1 : posb;
            posb_updated = idxa > idxb ? 1 : 0;
            posb += posb_updated;
            //printf ("the smaller index goes forward (after),  posa = %i, posb = %i\n", posa, posb);
        }
    }
    __syncthreads();


    for (int i = lane_id; i < blknnzctotal; i+= WARP_SIZE)
    {
                d_blkcsr_Col_C[nnzcstart + i] = s_blkcsr_Idx_C_local[i];
                d_blkcsr_Val_C[nnzcstart + i] = s_blkcsr_Val_C_local[i];
    }
/*
    if (lane_id < BLOCK_SIZE)
    {
        //unsigned int maskc = maskc;s_maskc_local[lane_id];
        int cnt = 0;
        int blknnzcstart = d_blkcsr_Ptr_C[global_warp_id * BLOCK_SIZE + lane_id];
        //int blknnzcstop = __shfl_down_sync(0xffffffff, blknnzcstart, 1);
        //blknnzcstop = (lane_id == BLOCK_SIZE - 1) ? blknnzcstop : blknnzctotal;
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++)
        //int i = 0;
        //while (cnt != blknnzcstop - blknnzcstart && i < BLOCK_SIZE)
        {
            int idx = ((maskc >> BLOCK_SIZE - i - 1) & 0x1) == 1 ? i : -1;
            if (idx != -1)
            {
                unsigned char col = idx;
                MAT_VAL_TYPE val = s_blkcsr_Val_C_local[lane_id * BLOCK_SIZE + idx];
                //MAT_VAL_TYPE val = s_blkcsr_Val_C_local[idx * BLOCK_SIZE + lane_id];
                d_blkcsr_Col_C[nnzcstart + blknnzcstart + cnt] = col;
                d_blkcsr_Val_C[nnzcstart + blknnzcstart + cnt] = val;
                cnt++;
            }
            //if (cnt == blknnzcstop - blknnzcstart) break;
            //i++;
        }
    }
*/
}

template <int SMEM_SIZE>
__global__
void stir_spgemm_step4_cuda_kernel_dns
                             (int *d_blkrowptrA,
                            const int* __restrict__ d_blkcolidxA,
                            int *d_nnzb_A,
                            MAT_VAL_TYPE *d_blkcsr_Val_A,
                            unsigned char *d_blkcsr_Col_A,
                            unsigned char *d_blkcsr_Ptr_A,
                            int blkmA, int blknA, int numblkA, int nnzA, 
                            int *d_blkcolptrB,
                            const int* __restrict__ d_blkrowidxB, 
                            int *d_nnzb_B,
                            MAT_VAL_TYPE *d_blkcsr_Val_B,
                            unsigned char *d_blkcsr_Col_B,
                            unsigned char *d_blkcsr_Ptr_B,
                            int blkmB, int blknB , int numblkB, int nnzB, 
                            int *d_blkrowidxC,
                            int *d_blkcolidxC,
                            unsigned char *d_blkcsr_Ptr_C,
                            unsigned char *d_blkcsr_Col_C,
                            MAT_VAL_TYPE *d_blkcsr_Val_C,
                            int *d_nnzb_C,
                            unsigned short *d_blkmaskC,
                            int numblkC,
                            int *d_blkid)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
     int global_warp_id = global_id / WARP_SIZE;

    if (global_warp_id >= numblkC) return;
    //global_warp_id = d_blkid[global_warp_id];

    const int nnzcstart = d_nnzb_C[global_warp_id];
    const int blknnzctotal = d_nnzb_C[global_warp_id+1] - nnzcstart;
    if (!blknnzctotal) return;

    const int local_warp_id = threadIdx.x / WARP_SIZE;
    __shared__ MAT_VAL_TYPE s_blkcsr_Val_C[WARP_PER_BLOCK * BLOCK_SIZE * BLOCK_SIZE];
    MAT_VAL_TYPE *s_blkcsr_Val_C_local = &s_blkcsr_Val_C[local_warp_id * BLOCK_SIZE * BLOCK_SIZE];

    __shared__ int s_matched_posa[WARP_PER_BLOCK * SMEM_SIZE];
    __shared__ int s_matched_posb[WARP_PER_BLOCK * SMEM_SIZE];
    __shared__ int s_matchedcnt[WARP_PER_BLOCK];

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int sublane_id = lane_id % (WARP_SIZE / BLOCK_SIZE);
    const int subwarp_id = lane_id / (WARP_SIZE / BLOCK_SIZE);
    //unsigned int maskc = 0;
    //__shared__ unsigned int s_maskc[WARP_PER_BLOCK * BLOCK_SIZE];
    //unsigned int *s_maskc_local = &s_maskc[local_warp_id * BLOCK_SIZE];

    int *s_matched_posa_local = &s_matched_posa[local_warp_id * SMEM_SIZE];
    int *s_matched_posb_local = &s_matched_posb[local_warp_id * SMEM_SIZE];
    int *s_matchedcnt_local = &s_matchedcnt[local_warp_id];


    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++)
        if (lane_id < BLOCK_SIZE)
            s_blkcsr_Val_C_local[i * BLOCK_SIZE + lane_id] = 0;
    if (!lane_id) 
        s_matchedcnt_local[0] = 0;

    //if (lane_id < BLOCK_SIZE)
    //    s_maskc_local[lane_id] = d_blkmaskC[global_warp_id * BLOCK_SIZE + lane_id];
    //__syncthreads();

    const int blki = d_blkrowidxC[global_warp_id];
    const int blkj = d_blkcolidxC[global_warp_id];

    const int abase = d_blkrowptrA[blki];
    const int astop = d_blkrowptrA[blki+1];
    const int astart = d_blkcolidxA[abase];
    const int aend = d_blkcolidxA[astop-1];
    int lena = d_blkrowptrA[blki+1] - abase;

    const int bbase = d_blkcolptrB[blkj];
    const int bstop = d_blkcolptrB[blkj+1];
    const int bstart = d_blkrowidxB[bbase];
    const int bend = d_blkrowidxB[bstop-1];
    int lenb = d_blkcolptrB[blkj+1] - bbase;


    if (lena < lenb)
    {
        for (int i = lane_id; i < lena; i+= WARP_SIZE)
        {
            int idxa = d_blkcolidxA[abase+i];
            int res = binary_search_exact_kernel(&d_blkrowidxB[bbase], 0, lenb-1, idxa);
            if (res != -1)
            {
                int pos = atomicAdd(s_matchedcnt_local, 1);
                if (pos < SMEM_SIZE)
                {
                    s_matched_posa_local[pos] = i;
                    s_matched_posb_local[pos] = res;
                }
            }
        }
    }
    else 
    {
        for (int i = lane_id; i < lenb; i+= WARP_SIZE)
        {
            int idxb = d_blkrowidxB[bbase+i];
            int res = binary_search_exact_kernel(&d_blkcolidxA[abase], 0, lena-1, idxb);
            if (res != -1)
            {
                int pos = atomicAdd(s_matchedcnt_local, 1);
                if (pos < SMEM_SIZE)
                {
                    s_matched_posa_local[pos] = res;
                    s_matched_posb_local[pos] = i;
                }
            }
        }
    }

    int matchedcnt = s_matchedcnt_local[0];




    if (matchedcnt <= SMEM_SIZE)
    {
        for (int i = 0; i < matchedcnt; i++)
        {
            int posa = s_matched_posa_local[i];
            int posb = s_matched_posb_local[i];

            const int nnzastart = d_nnzb_A[(abase+posa)];
            int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;
            unsigned char *d_csrRowPtrB = &d_blkcsr_Ptr_B[(bbase+posb)*BLOCK_SIZE];
            const int nnzbstart = d_nnzb_B[(bbase+posb)];
            int nnztotalb = d_nnzb_B[(bbase+posb) + 1] - nnzbstart;

            for (int i = lane_id; i < nnztotala; i+= WARP_SIZE)
            {
                unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+i];
                int rowidxa = rowcolidx >> 4;
                int rowidxb = rowcolidx & 0xf;
                MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart+i];

                const int startb = d_csrRowPtrB[rowidxb];
                const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : d_csrRowPtrB[rowidxb+1];
                for (int k = startb; k < stopb; k++)
                {
                    //unsigned char colidx_packet = d_csrColIdxB[k/2];
                    //unsigned char colidx = (k % 2 == 0) ? (colidx >> 4 & 0x0F) : (colidx & 0x0F);
                    unsigned char colidx = d_blkcsr_Col_B[nnzbstart+k];
                    MAT_VAL_TYPE valb = d_blkcsr_Val_B[nnzbstart+k];
                    //maskc = maskc | (0x1 << (BLOCK_SIZE - colidx - 1));
                    //atomicOr(&s_maskc_local[rowidxa], (unsigned int)(0x1 << (BLOCK_SIZE - colidx - 1)));
                    //s_dnsidx_local[subwarp_id * BLOCK_SIZE + colidx] = 1;
                    //s_blkcsr_Val_C_local[subwarp_id * BLOCK_SIZE + colidx] += val * valb;
                    atomicAdd(&s_blkcsr_Val_C_local[rowidxa * BLOCK_SIZE + colidx], val * valb);
                    //atomicAdd(&s_blkcsr_Val_C_local[colidx * BLOCK_SIZE + rowidxa], val * valb);
                }
            }
        }
    }
    else 
    {
        int posa_real = 0;
        int posb_real = 0;
        if (bstart > astart)
        {
            //posa_real = binary_search_right_boundary_kernel((lena <= 128 ? s_blkcolidxA : &d_blkcolidxA[abase]), bstart, lena);
            int posa_real_new = binary_search_right_boundary_kernel(&d_blkcolidxA[abase], bstart, lena);
            posa_real = posa_real_new < 0 ? 0 : posa_real_new;
        }
        else if (bstart < astart)
        {
            int posb_real_new = binary_search_right_boundary_kernel(&d_blkrowidxB[bbase], astart, lenb);
            posb_real = posb_real_new < 0 ? 0 : posb_real_new;
        }

        if (bstop < astop)
        {
            int lena_new = binary_search_right_boundary_kernel(&d_blkcolidxA[abase], bend, lena) + 1;
            lena = lena_new > lena ? lena : lena_new;
        }
        else if (bstop > astop)
        {
            int lenb_new = binary_search_right_boundary_kernel(&d_blkrowidxB[bbase], aend, lenb) + 1;
            lenb = lenb_new > lenb ? lenb : lenb_new;
        }

        int posa = posa_real;
        int posb = posb_real;
        int idxa = 0;
        int idxb = 0;
        int posa_updated = 1;
        int posb_updated = 1;

        while(posa < lena && posb < lenb)
        {
            idxa = posa_updated ? d_blkcolidxA[abase+posa] : idxa; //a[posa] : idxa;
            //idxa = posa_updated ? (lena < 128 ? s_blkcolidxA[posa] : d_blkcolidxA[abase+posa]) : idxa; //a[posa] : idxa;
            idxb = posb_updated ? d_blkrowidxB[bbase+posb] : idxb; //b[posb] : idxb;

            if (idxa == idxb)
            {
                //atomicAdd(&s_numblkC[0], 1);
                //int posc = atomicAdd(&d_blkrowptrC_offset[blki], 1);
                //d_blkrowidxC[posc] = blki;
                //d_blkcolidxC[posc] = blkj;
                //atomicAdd(&d_blkrowptrC[blki], 1);
                //break;

    //if (1)
    {
    /*
                // do one pair of SpGEMM
                //const int starta = d_blkcsr_Ptr_A[(abase+posa)*BLOCK_SIZE] + sublane_id;
                unsigned char *d_csrRowPtrA = &d_blkcsr_Ptr_A[(abase+posa)*BLOCK_SIZE];
                const int nnzastart = d_nnzb_A[(abase+posa)];
                int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;
                const int starta = d_csrRowPtrA[subwarp_id];
                const int stopa = (subwarp_id == BLOCK_SIZE - 1) ? nnztotala : d_csrRowPtrA[subwarp_id+1];

                unsigned char *d_csrRowPtrB = &d_blkcsr_Ptr_B[(bbase+posb)*BLOCK_SIZE];
                const int nnzbstart = d_nnzb_B[(bbase+posb)];
                int nnztotalb = d_nnzb_B[(bbase+posb) + 1] - nnzbstart;
                //const int starta = d_csrRowPtrA[subwarp_id];
            // const int stopa = (subwarp_id = BLOCK_SIZE - 1) ? nnztotala : d_csrRowPtrA[subwarp_id+1];

                for (int j = starta + sublane_id; j < stopa; j += 2)
                {
                    int rowidxb = d_blkcsr_Col_A[nnzastart+j] & 0xf;
                    MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart+j];
                    const int startb = d_csrRowPtrB[rowidxb];
                    const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : d_csrRowPtrB[rowidxb+1];
                    for (int k = startb; k < stopb; k++)
                    {
                        //unsigned char colidx_packet = d_csrColIdxB[k/2];
                        //unsigned char colidx = (k % 2 == 0) ? (colidx >> 4 & 0x0F) : (colidx & 0x0F);
                        unsigned char colidx = d_blkcsr_Col_B[nnzbstart+k];
                        MAT_VAL_TYPE valb = d_blkcsr_Val_B[nnzbstart+k];
                        maskc = maskc | (0x1 << (BLOCK_SIZE - colidx - 1));
                        //s_dnsidx_local[subwarp_id * BLOCK_SIZE + colidx] = 1;
                        //s_blkcsr_Val_C_local[subwarp_id * BLOCK_SIZE + colidx] += val * valb;
    //                    atomicAdd(&s_blkcsr_Val_C_local[subwarp_id * BLOCK_SIZE + colidx], val * valb);
                    }
                }
    */
    //}
    //else
    //{
                const int nnzastart = d_nnzb_A[(abase+posa)];
                int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;
                unsigned char *d_csrRowPtrB = &d_blkcsr_Ptr_B[(bbase+posb)*BLOCK_SIZE];
                const int nnzbstart = d_nnzb_B[(bbase+posb)];
                int nnztotalb = d_nnzb_B[(bbase+posb) + 1] - nnzbstart;

                for (int i = lane_id; i < nnztotala; i+= WARP_SIZE)
                {
                    unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+i];
                    int rowidxa = rowcolidx >> 4;
                    int rowidxb = rowcolidx & 0xf;
                    MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart+i];

                    const int startb = d_csrRowPtrB[rowidxb];
                    const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : d_csrRowPtrB[rowidxb+1];
                    for (int k = startb; k < stopb; k++)
                    {
                        //unsigned char colidx_packet = d_csrColIdxB[k/2];
                        //unsigned char colidx = (k % 2 == 0) ? (colidx >> 4 & 0x0F) : (colidx & 0x0F);
                        unsigned char colidx = d_blkcsr_Col_B[nnzbstart+k];
                        MAT_VAL_TYPE valb = d_blkcsr_Val_B[nnzbstart+k];
                        //maskc = maskc | (0x1 << (BLOCK_SIZE - colidx - 1));
                        //atomicOr(&s_maskc_local[rowidxa], (unsigned int)(0x1 << (BLOCK_SIZE - colidx - 1)));
                        //s_dnsidx_local[subwarp_id * BLOCK_SIZE + colidx] = 1;
                        //s_blkcsr_Val_C_local[subwarp_id * BLOCK_SIZE + colidx] += val * valb;
                        atomicAdd(&s_blkcsr_Val_C_local[rowidxa * BLOCK_SIZE + colidx], val * valb);
                        //atomicAdd(&s_blkcsr_Val_C_local[colidx * BLOCK_SIZE + rowidxa], val * valb);
                    }
                }
    }
                // do spgemm of this pair
                //printf ("do spgemm of this pair, idxa = %i, idxb = %i\n", idxa, idxb);
                posa++;
                posa_updated = 1;
                posb++;
                posb_updated = 1;
            }
            else
            {
                // the smaller index goes forward
                //printf ("the smaller index goes forward (before), posa = %i, posb = %i\n", posa, posb);
                //posa = idxa < idxb ? posa + 1 : posa;
                posa_updated = idxa < idxb ? 1 : 0;
                posa += posa_updated;
                //posb = idxa > idxb ? posb + 1 : posb;
                posb_updated = idxa > idxb ? 1 : 0;
                posb += posb_updated;
                //printf ("the smaller index goes forward (after),  posa = %i, posb = %i\n", posa, posb);
            }
        }
        //__syncthreads();
    }

//    maskc |= __shfl_down_sync(0xffffffff, maskc, 1);
//maskc = __shfl_sync(0xffffffff, maskc, lane_id * 2);
/*
    //int nnzcnt = 0;
    //if (lane_id < BLOCK_SIZE) 
    //    nnzcnt = __popc(maskc);
    int nnzcnt = lane_id % 2 == 0 ? __popc(maskc) : 0;

    nnzcnt = __shfl_sync(0xffffffff, nnzcnt, lane_id * 2);

    nnzcnt = lane_id < BLOCK_SIZE ? nnzcnt : 0;

    //nnzcnt = sum_32_shfl(nnzcnt);
    
    int nnzcnt_scan = scan_32_shfl(nnzcnt, lane_id);
    //if (lane_id < BLOCK_SIZE) 
    //    d_blkcsr_Ptr_C[global_warp_id * BLOCK_SIZE + lane_id] = nnzcnt_scan - nnzcnt;

    nnzcnt = __shfl_sync(0xffffffff, nnzcnt_scan, BLOCK_SIZE);
    //*s_blknnzC = nnzcnt;
*/
    //int dnscnt = 0;
    //if (lane_id < BLOCK_SIZE)
    //{
    //    for (int i = 0; i < BLOCK_SIZE; i++)
    //        dnscnt = s_dnsidx_local[i * BLOCK_SIZE + lane_id] == 1 ? dnscnt + 1 : dnscnt;
    //}
    //__syncthreads();
    //dnscnt = sum_32_shfl(dnscnt);
    //__syncthreads();

    if (blknnzctotal == 256)
    {
        for (int i = lane_id; i < BLOCK_SIZE * BLOCK_SIZE; i += WARP_SIZE)
        {
            d_blkcsr_Col_C[nnzcstart + i] = i % BLOCK_SIZE;
            d_blkcsr_Val_C[nnzcstart + i] = s_blkcsr_Val_C_local[i];
        }
    }

    //unsigned short maskc = lane_id < BLOCK_SIZE ? d_blkmaskC[global_warp_id * BLOCK_SIZE + lane_id] : 0; //s_maskc_local[lane_id];

    if (blknnzctotal != 256 && lane_id < BLOCK_SIZE)
    {
        unsigned short maskc = d_blkmaskC[global_warp_id * BLOCK_SIZE + lane_id]; //s_maskc_local[lane_id];
        int cnt = 0;
        int blknnzcstart = d_blkcsr_Ptr_C[global_warp_id * BLOCK_SIZE + lane_id];
        //int blknnzcstop = __shfl_down_sync(0xffffffff, blknnzcstart, 1);
        //blknnzcstop = (lane_id == BLOCK_SIZE - 1) ? blknnzcstop : blknnzctotal;
        #pragma unroll 16
        for (int i = 0; i < BLOCK_SIZE; i++)
        //int i = 0;
        //while (cnt != blknnzcstop - blknnzcstart && i < BLOCK_SIZE)
        {
            int idx = ((maskc >> BLOCK_SIZE - i - 1) & 0x1) == 1 ? i : -1;
            if (idx != -1)
            {
                //unsigned char col = idx;
                //MAT_VAL_TYPE val = s_blkcsr_Val_C_local[lane_id * BLOCK_SIZE + idx];
                //MAT_VAL_TYPE val = s_blkcsr_Val_C_local[idx * BLOCK_SIZE + lane_id];
                d_blkcsr_Col_C[nnzcstart + blknnzcstart + cnt] = idx;
                d_blkcsr_Val_C[nnzcstart + blknnzcstart + cnt] = s_blkcsr_Val_C_local[lane_id * BLOCK_SIZE + idx];
                cnt++;
            }
            //if (cnt == blknnzcstop - blknnzcstart) break;
            //i++;
        }
    }

}

void step4 (int *d_blkrowptrA, int *d_blkcolidxA, int blkmA, int blknA,int *nnzb_A ,int mA,
            MAT_VAL_TYPE *blkcsr_Val_A , unsigned char  *blkcsr_Col_A , unsigned char *blkcsr_Ptr_A ,
            int *d_blkcolptrB, int *d_blkrowidxB, int blkmB, int blknB , int *nnzb_B ,
            MAT_VAL_TYPE *blkcsr_Val_B , unsigned char  *blkcsr_Col_B , unsigned char *blkcsr_Ptr_B ,
            int *d_blkrowptrC, int *d_blkcolidxC,int *nnzb_C ,
            MAT_VAL_TYPE *blkcsr_Val_C , unsigned char  *blkcsr_Col_C , unsigned char *blkcsr_Ptr_C )

{
    MAT_VAL_TYPE *blkcval = (MAT_VAL_TYPE *)malloc((BLOCK_SIZE * BLOCK_SIZE) *sizeof(MAT_VAL_TYPE));
  
    char * blkc = (char *)malloc((BLOCK_SIZE * BLOCK_SIZE) *sizeof(char));
    for (int blki =0 ;blki <blkmA ;blki++)
    {
        int rowlen = blki == blkmA -1 ? mA- (blkmA -1 ) * BLOCK_SIZE : BLOCK_SIZE ;
        for (int blkj =d_blkrowptrC[blki]; blkj <d_blkrowptrC[blki + 1]; blkj++)
        {
            int count =0;
            int blkccolid = d_blkcolidxC[blkj];
        //    int rowlen = blki == blkmA -1 ? mA- (blkmA -1 ) * BLOCK_SIZE : BLOCK_SIZE ;
        //    int collen = blkccolid == blknB -1 ? nB - (blknB -1) *BLOCK_SIZE : BLOCK_SIZE ;
            memset (blkc , 0, (BLOCK_SIZE * BLOCK_SIZE) *sizeof(char));
            memset (blkcval,0,(BLOCK_SIZE * BLOCK_SIZE) *sizeof(MAT_VAL_TYPE));
        //    memset(blkccol,0,(BLOCK_SIZE * BLOCK_SIZE) *sizeof(char)) ;

            int posA = d_blkrowptrA[blki];
            int posB = d_blkcolptrB[blkccolid];
            int idxA= 0;
            int idxB =0;
            int posa_updated =1;
            int posb_updated =1;
            while (posA < d_blkrowptrA[blki +1] && posB <d_blkcolptrB[blkccolid + 1])
            {
                idxA = posa_updated ? d_blkcolidxA[posA] : idxA ;
                idxB = posb_updated ? d_blkrowidxB[posB] : idxB ;
                if (idxA == idxB)  // do spgemm of this pair
                {
                //        printf ("do spgemm of this pair, idxa = %i, idxb = %i\n", idxA, idxB);
                //for each row of block
                    for (int ri =0;ri <BLOCK_SIZE ;ri ++ )
                    {
                        if (ri == rowlen)
                            break;
                        int stopa = ri == BLOCK_SIZE -1 ? nnzb_A[posA +1] - nnzb_A[posA] : blkcsr_Ptr_A[posA * BLOCK_SIZE + ri + 1] ;
                
                        for (int i=blkcsr_Ptr_A[ posA * BLOCK_SIZE+ ri];i<stopa;i++)
                        {
                            int cola= blkcsr_Col_A[nnzb_A[posA]+i] ;
                            int stopb = cola == BLOCK_SIZE -1  ? nnzb_B[posB +1]- nnzb_B[posB] : blkcsr_Ptr_B[posB * BLOCK_SIZE+cola +1] ;
                            for (int bi= blkcsr_Ptr_B[posB * BLOCK_SIZE +cola ];bi< stopb; bi++)
                            {
                                const int colb = blkcsr_Col_B[nnzb_B[posB] + bi];

                                blkcval[ri * BLOCK_SIZE + colb] += blkcsr_Val_A[nnzb_A[posA]+i] * blkcsr_Val_B[nnzb_B[posB] + bi] ;
                                if (blkc[ri * BLOCK_SIZE + colb] == 0)
                                {
                                    blkc[ri * BLOCK_SIZE + colb] = 1;
                                }
                            }

                        }
                    }
                    posA++;
                    posa_updated = 1;
                    posB++;
                    posb_updated = 1;
                }
                else 
                {
                //    printf ("the smaller index goes forward (before), posa = %i, posb = %i\n", posA, posA);
                    posA = idxA < idxB ? posA + 1 : posA;
                    posa_updated = idxA < idxB ? 1 : 0;
                    posB = idxA > idxB ? posB + 1 : posB;
                    posb_updated = idxA > idxB ? 1 : 0;
                //    printf ("the smaller index goes forward (after),  posa = %i, posb = %i\n", posA, posA);
                }

            }  
        /*    for (int ci=0;ci< BLOCK_SIZE * BLOCK_SIZE ; ci ++)
            {
                if (blkc[ci]== 1)
                {
                    count ++ ;
                }
            }
            nnzb_C[blkj] = count ;
        */
        //    printf("count = %d\n",count);
            for (int ri =0;ri < BLOCK_SIZE ;ri ++)
            {
                for (int ci =0;ci <BLOCK_SIZE ;ci ++)
                {
                    if (blkc[ri * BLOCK_SIZE +ci] == 1)
                    {
                        blkcsr_Val_C[nnzb_C[blkj] + count ] = blkcval[ri * BLOCK_SIZE +ci];
                        blkcsr_Col_C[nnzb_C[blkj] + count ] = ci ;
                        count ++ ;
                    }
                }
                if (ri < BLOCK_SIZE -1)
                    blkcsr_Ptr_C[BLOCK_SIZE * blkj + ri +1] = count ;
            }

        }
    }


}


void step4_cuda (int *blkrowptrA, int *blkcolidxA, int blkmA, int blknA, int numblkA, int nnzA, int *nnzb_A ,int mA,
            MAT_VAL_TYPE *blkcsr_Val_A , unsigned char  *blkcsr_Col_A , unsigned char *blkcsr_Ptr_A ,
            int *blkcolptrB, int *blkrowidxB, int blkmB, int blknB , int numblkB, int nnzB, int *nnzb_B ,
            MAT_VAL_TYPE *blkcsr_Val_B , unsigned char  *blkcsr_Col_B , unsigned char *blkcsr_Ptr_B ,
            int *blkrowptrC, int *blkcolidxC,
            int *d_blkrowidxC, int *d_blkcolidxC, 
            unsigned char *d_blkcsr_Ptr_C, unsigned char *d_blkcsr_Col_C, MAT_VAL_TYPE *d_blkcsr_Val_C, unsigned short *d_blkmaskC,
            int *nnzb_C,  int numblkC, 
            MAT_VAL_TYPE *blkcsr_Val_C_golden, 
            unsigned char *blkcsr_Col_C_golden, 
            unsigned char *blkcsr_Ptr_C_golden, int nnzC, 
int *d_blksmem_cnt, int *d_blkdns_cnt, int *d_blkid_smem, int *d_blkid_dns)
{


    int *d_blkrowptrA;
    int *d_blkcolidxA; 
    int *d_nnzb_A; 
    MAT_VAL_TYPE *d_blkcsr_Val_A;
    unsigned char *d_blkcsr_Col_A;
    unsigned char *d_blkcsr_Ptr_A;

    cudaMalloc((void **)&d_blkrowptrA, (blkmA+1) * sizeof(int));
    cudaMalloc((void **)&d_blkcolidxA, numblkA * sizeof(int));
    cudaMalloc((void **)&d_nnzb_A, (numblkA+1) * sizeof(int));
    cudaMalloc((void **)&d_blkcsr_Val_A, nnzA * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_blkcsr_Col_A, nnzA * sizeof(unsigned char));
    cudaMalloc((void **)&d_blkcsr_Ptr_A, numblkA * BLOCK_SIZE  * sizeof(unsigned char));

    cudaMemcpy(d_blkrowptrA,     blkrowptrA,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcolidxA,     blkcolidxA,     numblkA * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnzb_A,     nnzb_A,     (numblkA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Val_A,     blkcsr_Val_A,     nnzA * sizeof(MAT_VAL_TYPE),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Col_A,     blkcsr_Col_A,     nnzA * sizeof(unsigned char),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Ptr_A,     blkcsr_Ptr_A,     numblkA * BLOCK_SIZE * sizeof(unsigned char),              cudaMemcpyHostToDevice);

    int *d_blkcolptrB;
    int *d_blkrowidxB; 
    int *d_nnzb_B; 
    MAT_VAL_TYPE *d_blkcsr_Val_B;
    unsigned char *d_blkcsr_Col_B;
    unsigned char *d_blkcsr_Ptr_B;

    cudaMalloc((void **)&d_blkcolptrB, (blknB+1) * sizeof(int));
    cudaMalloc((void **)&d_blkrowidxB, numblkB * sizeof(int));
    cudaMalloc((void **)&d_nnzb_B, (numblkB+1) * sizeof(int));
    cudaMalloc((void **)&d_blkcsr_Val_B, nnzB * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_blkcsr_Col_B, nnzB * sizeof(unsigned char));
    cudaMalloc((void **)&d_blkcsr_Ptr_B, numblkB * BLOCK_SIZE  * sizeof(unsigned char));

    cudaMemcpy(d_blkcolptrB,     blkcolptrB,     (blknB+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkrowidxB,     blkrowidxB,     numblkB * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnzb_B,     nnzb_B,     (numblkB+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Val_B,     blkcsr_Val_B,     nnzB * sizeof(MAT_VAL_TYPE),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Col_B,     blkcsr_Col_B,     nnzB * sizeof(unsigned char),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Ptr_B,     blkcsr_Ptr_B,     numblkB * BLOCK_SIZE * sizeof(unsigned char),              cudaMemcpyHostToDevice);


    //int *d_blkrowptrC;
    //int *d_blkrowidxC;
    //int *d_blkcolidxC;
    int *d_nnzb_C;

    //cudaMalloc((void **)&d_blkrowptrC, (blkmA+1) * sizeof(int));
    //cudaMalloc((void **)&d_blkrowidxC, numblkC * sizeof(int));
    //cudaMalloc((void **)&d_blkcolidxC, numblkC * sizeof(int));
    cudaMalloc((void **)&d_nnzb_C, (numblkC+1) * sizeof(int));

    //cudaMemcpy(d_blkrowptrC,     blkrowptrC,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemcpy(d_blkrowidxC,     blkrowidxC,     numblkC * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemcpy(d_blkcolidxC,     blkcolidxC,     numblkC * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemset(d_nnzb_C, 0, numblkC * sizeof(int));
cudaMemcpy(d_nnzb_C,     nnzb_C,     (numblkC+1) * sizeof(int),              cudaMemcpyHostToDevice);

    struct timeval t1, t2;

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
/*
int blksmem_cnt = 0;
int blkdns_cnt = 0;
cudaMemcpy(&blksmem_cnt,     d_blksmem_cnt,     sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(&blkdns_cnt,     d_blkdns_cnt,     sizeof(int), cudaMemcpyDeviceToHost);

    int num_threads = WARP_SIZE * WARP_PER_BLOCK;
    int num_blocks = ceil((double)blksmem_cnt/(double)WARP_PER_BLOCK);
    stir_spgemm_step4_cuda_kernel_smem<32><<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, 
d_blkcsr_Col_C,d_blkcsr_Val_C,
d_nnzb_C, d_blkmaskC, blksmem_cnt, d_blkid_smem);

    num_blocks = ceil((double)blkdns_cnt/(double)WARP_PER_BLOCK);
    stir_spgemm_step4_cuda_kernel_dns<<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, 
d_blkcsr_Col_C,d_blkcsr_Val_C,
d_nnzb_C, blkdns_cnt, d_blkid_dns);
*/
    int num_threads = WARP_SIZE * WARP_PER_BLOCK;
    int num_blocks = ceil((double)numblkC/(double)WARP_PER_BLOCK);
    stir_spgemm_step4_cuda_kernel_dns<64><<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, 
d_blkcsr_Col_C,d_blkcsr_Val_C,
d_nnzb_C, d_blkmaskC, numblkC, NULL);


    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    double time_kernel = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("CUDA step4 kernel = %.2f ms\n", time_kernel);


    MAT_VAL_TYPE *h_blkcsr_Val_C = (MAT_VAL_TYPE *)malloc(nnzC*sizeof(MAT_VAL_TYPE));
    cudaMemcpy(h_blkcsr_Val_C,     d_blkcsr_Val_C,     nnzC*sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);

    unsigned char *h_blkcsr_Col_C = (unsigned char *)malloc(nnzC*sizeof(unsigned char));
    cudaMemcpy(h_blkcsr_Col_C,     d_blkcsr_Col_C,     nnzC*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    unsigned char *h_blkcsr_Ptr_C = (unsigned char *)malloc(BLOCK_SIZE*numblkC*sizeof(unsigned char));
    cudaMemcpy(h_blkcsr_Ptr_C,     d_blkcsr_Ptr_C,     BLOCK_SIZE*numblkC*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    int errcnt = 0;
    for (int i = 0; i < BLOCK_SIZE*numblkC; i++)
        if (h_blkcsr_Ptr_C[i] != blkcsr_Ptr_C_golden[i])
            {errcnt++;}
    printf("step 4, h_blkcsr_Ptr_C, errcnt = %i\n", errcnt);

    errcnt = 0;
    for (int i = 0; i < nnzC; i++)
        if (h_blkcsr_Col_C[i] != blkcsr_Col_C_golden[i])
            {errcnt++;}
    printf("step 4, h_blkcsr_Col_C, errcnt = %i\n", errcnt);

    errcnt = 0;
    for (int i = 0; i < nnzC; i++)
        if (h_blkcsr_Val_C[i] != blkcsr_Val_C_golden[i])
            {//printf("%f, %f\n", h_blkcsr_Val_C[i], blkcsr_Val_C_golden[i]); 
            errcnt++;}
    printf("step 4, h_blkcsr_Val_C, errcnt = %i\n", errcnt);

    cudaFree(d_blkrowptrA);
    cudaFree(d_blkcolidxA); 
    cudaFree(d_nnzb_A); 
    cudaFree(d_blkcsr_Val_A);
    cudaFree(d_blkcsr_Col_A);
    cudaFree(d_blkcsr_Ptr_A);

    cudaFree(d_blkcolptrB);
    cudaFree(d_blkrowidxB); 
    cudaFree(d_nnzb_B); 
    cudaFree(d_blkcsr_Val_B);
    cudaFree(d_blkcsr_Col_B);
    cudaFree(d_blkcsr_Ptr_B);

    //cudaFree(d_blkrowptrC);
    //cudaFree(d_blkrowidxC);
    //cudaFree(d_blkcolidxC);
    cudaFree(d_nnzb_C);
    free(h_blkcsr_Ptr_C);
    free(h_blkcsr_Col_C);
    free(h_blkcsr_Val_C);
}


void stepall_cuda_new (int *blkrowptrA, int *blkcolidxA, int blkmA, int blknA, int numblkA, int nnzA, int *nnzb_A ,int mA,
            MAT_VAL_TYPE *blkcsr_Val_A , unsigned char  *blkcsr_Col_A , unsigned char *blkcsr_Ptr_A , unsigned short *blkmaskA,
            int *blkcolptrB, int *blkrowidxB, int *blkrowptrB, int *blkcolidxB, 
            int blkmB, int blknB , int numblkB, int nnzB, int *nnzb_B ,
            MAT_VAL_TYPE *blkcsr_Val_B , unsigned char  *blkcsr_Col_B , unsigned char *blkcsr_Ptr_B , unsigned short *blkmaskB,
            int *blkrowptrC_golden, int *blkcolidxC_golden,
            int *nnzb_C_golden,  int numblkC_golden, 
            MAT_VAL_TYPE *blkcsr_Val_C_golden, 
            unsigned char *blkcsr_Col_C_golden, 
            unsigned char *blkcsr_Ptr_C_golden, int nnzC_golden,unsigned long long int nnzCub, 
unsigned long long int *nnzC_computed, double *compression_rate, double *time_stir, double *gflops_stir)
{
//for (int i = 0; i < blkmA; i++)
//    printf("[%i],%i\n", i, blkrowptrA[i+1] - blkrowptrA[i]);



    int *d_blkrowptrA;
    int *d_blkcolidxA; 
    int *d_nnzb_A; 
    MAT_VAL_TYPE *d_blkcsr_Val_A;
    unsigned char *d_blkcsr_Col_A;
    unsigned char *d_blkcsr_Ptr_A;

    cudaMalloc((void **)&d_blkrowptrA, (blkmA+1) * sizeof(int));
    cudaMalloc((void **)&d_blkcolidxA, numblkA * sizeof(int));
    cudaMalloc((void **)&d_nnzb_A, (numblkA+1) * sizeof(int));
    cudaMalloc((void **)&d_blkcsr_Val_A, nnzA * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_blkcsr_Col_A, nnzA * sizeof(unsigned char));
    cudaMalloc((void **)&d_blkcsr_Ptr_A, numblkA * BLOCK_SIZE  * sizeof(unsigned char));

    cudaMemcpy(d_blkrowptrA,     blkrowptrA,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcolidxA,     blkcolidxA,     numblkA * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnzb_A,     nnzb_A,     (numblkA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Val_A,     blkcsr_Val_A,     nnzA * sizeof(MAT_VAL_TYPE),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Col_A,     blkcsr_Col_A,     nnzA * sizeof(unsigned char),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Ptr_A,     blkcsr_Ptr_A,     numblkA * BLOCK_SIZE * sizeof(unsigned char),              cudaMemcpyHostToDevice);

    int *d_blkcolptrB;
    int *d_blkrowidxB; 
    int *d_blkrowptrB; 
    int *d_blkcolidxB;
    int *d_nnzb_B; 
    MAT_VAL_TYPE *d_blkcsr_Val_B;
    unsigned char *d_blkcsr_Col_B;
    unsigned char *d_blkcsr_Ptr_B;

    cudaMalloc((void **)&d_blkcolptrB, (blknB+1) * sizeof(int));
    cudaMalloc((void **)&d_blkrowidxB, numblkB * sizeof(int));
    cudaMalloc((void **)&d_blkrowptrB, (blkmB+1) * sizeof(int));
    cudaMalloc((void **)&d_blkcolidxB, numblkB * sizeof(int));
    cudaMalloc((void **)&d_nnzb_B, (numblkB+1) * sizeof(int));
    cudaMalloc((void **)&d_blkcsr_Val_B, nnzB * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_blkcsr_Col_B, nnzB * sizeof(unsigned char));
    cudaMalloc((void **)&d_blkcsr_Ptr_B, numblkB * BLOCK_SIZE  * sizeof(unsigned char));

    cudaMemcpy(d_blkcolptrB,     blkcolptrB,     (blknB+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkrowidxB,     blkrowidxB,     numblkB * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkrowptrB,     blkrowptrB,     (blkmB+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcolidxB,     blkcolidxB,     numblkB * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnzb_B,     nnzb_B,     (numblkB+1) * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Val_B,     blkcsr_Val_B,     nnzB * sizeof(MAT_VAL_TYPE),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Col_B,     blkcsr_Col_B,     nnzB * sizeof(unsigned char),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Ptr_B,     blkcsr_Ptr_B,     numblkB * BLOCK_SIZE * sizeof(unsigned char),              cudaMemcpyHostToDevice);

    unsigned short *d_blkmaskB;
    cudaMalloc((void **)&d_blkmaskB, numblkB * BLOCK_SIZE  * sizeof(unsigned short));
    cudaMemcpy(d_blkmaskB,     blkmaskB,     numblkB * BLOCK_SIZE * sizeof(unsigned short),              cudaMemcpyHostToDevice);

    unsigned short *d_blkmaskA;
    cudaMalloc((void **)&d_blkmaskA, numblkA * BLOCK_SIZE  * sizeof(unsigned short));
    cudaMemcpy(d_blkmaskA,     blkmaskA,     numblkA * BLOCK_SIZE * sizeof(unsigned short),              cudaMemcpyHostToDevice);

double stir_spgemm_time = 0;
for (int ri = 0; ri < REPEAT_NUM; ri++)
{
    // call cuda kernel
    struct timeval t1, t2;
    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

//int *d_blkrowptrA;
// int *d_blkcolidxA; 
            
            int *d_blkrowptrC;

    //cudaMalloc((void **)&d_blkrowptrA, (blkmA+1) * sizeof(int));
    //cudaMalloc((void **)&d_blkcolidxA, numblkA * sizeof(int));


    cudaMalloc((void **)&d_blkrowptrC, (blkmA+1) * sizeof(int));
  //  cudaMemset(d_blkrowptrC, 0, (blkmA+1) * sizeof(int));

    //cudaMemcpy(d_blkrowptrA,     blkrowptrA,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemcpy(d_blkcolidxA,     blkcolidxA,     numblkA * sizeof(int),              cudaMemcpyHostToDevice);






/*
    int num_threads = 64;
    int num_blocks = blkmA;
    stir_spgemm_step1_cuda_kernel<<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, blkmA, 
                              d_blkcolptrB, d_blkrowidxB, blknB, d_blkrowptrC);

exclusive_scan_device_cuda(d_blkrowptrC, blkmA+1);
int nbc = 0;
cudaMemcpy(&nbc,     &d_blkrowptrC[blkmA],     sizeof(int), cudaMemcpyDeviceToHost);
*numblkC = nbc;
*/
sfBIN bin;
    /* Initialize bin */
    init_bin(&bin, blkmA);

    /* Set max bin */
    //set_max_bin(a->d_rpt, a->d_col, b->d_rpt, &bin, M);
    set_max_bin(d_blkrowptrA, d_blkcolidxA, d_blkrowptrB, &bin, blkmA);

    //    cudaMalloc((void **)&d_csrRowPtrC, (blkmA+1) * sizeof(int));
int numblkC = 0;
    /* Count nz of C */
    set_row_nnz(d_blkrowptrA, d_blkcolidxA,
                d_blkrowptrB, d_blkcolidxB,
                d_blkrowptrC,
                &bin,
                blkmA,
                &numblkC);

   //printf("nsparse nnzC = %i\n", nnzC);

    /* Set bin */
    set_min_bin(&bin, blkmA);

        //spgemm_nsparse_executor_step1(mA, nA,  nnzA, d_csrRowPtrA, d_csrColIdxA, 
        //                              mB, nB,  nnzB, d_csrRowPtrB, d_csrColIdxB, 
        //                              mC, nC, &nnzC, d_csrRowPtrC);

     //   cudaMalloc((void **)&d_csrColIdxC, *numblkC * sizeof(int));

//    cudaDeviceSynchronize();
//    gettimeofday(&t2, NULL);

 //   double time_kernel = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
//    printf("CUDA step1 new kernel = %.2f ms, numblkC = %i\n", time_kernel, *numblkC);



    //cudaFree(d_blkrowptrA);
    //cudaFree(d_blkcolidxA);
    //cudaFree(d_blkrowptrB);
    //cudaFree(d_blkcolidxB);
    //cudaFree(d_blkrowptrC);
    //free(h_blkrowptrC);

   // printf("step 1 success!\n");

    int *d_blkrowidxC;
    cudaMalloc((void **)&d_blkrowidxC, numblkC * sizeof(int));
    int *d_blkcolidxC;
    cudaMalloc((void **)&d_blkcolidxC, numblkC * sizeof(int));
    unsigned short *d_blkmaskC;
    cudaMalloc((void **)&d_blkmaskC, numblkC *BLOCK_SIZE * sizeof(unsigned short));
    //cudaMemset(d_blkmaskC, 0, numblkC *BLOCK_SIZE * sizeof(unsigned short));

//int *d_blkrowptrA;
 //int *d_blkcolidxA; 
         //   int *d_blkrowptrB; int *d_blkcolidxB;
          //  int *d_blkrowptrC;
   //         int *d_blkrowptrC_offset;

    //cudaMalloc((void **)&d_blkrowptrA, (blkmA+1) * sizeof(int));
    //cudaMalloc((void **)&d_blkcolidxA, numblkA * sizeof(int));
  //  cudaMalloc((void **)&d_blkrowptrB, (blkmB+1) * sizeof(int));
   // cudaMalloc((void **)&d_blkcolidxB, numblkB * sizeof(int));
  //  cudaMalloc((void **)&d_blkrowptrC, (blkmA+1) * sizeof(int));
  //  cudaMalloc((void **)&d_blkrowptrC_offset, (blkmA+1) * sizeof(int));

    //cudaMemcpy(d_blkrowptrA,     blkrowptrA,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemcpy(d_blkcolidxA,     blkcolidxA,     numblkA * sizeof(int),              cudaMemcpyHostToDevice);
  //  cudaMemcpy(d_blkrowptrB,     blkrowptrB,     (blkmB+1) * sizeof(int),              cudaMemcpyHostToDevice);
  //  cudaMemcpy(d_blkcolidxB,     blkcolidxB,     numblkB * sizeof(int),              cudaMemcpyHostToDevice);
   // cudaMemcpy(d_blkrowptrC,     blkrowptrC,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
  //  cudaMemcpy(d_blkrowptrC_offset,     d_blkrowptrC,     (blkmA+1) * sizeof(int),              cudaMemcpyDeviceToDevice);

    // call cuda kernel
  //  struct timeval t1, t2;

  //  cudaDeviceSynchronize();
 //   gettimeofday(&t1, NULL);
/*
    int num_threads = 64;
    int num_blocks = blkmA;
    stir_spgemm_step2_cuda_kernel<<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, blkmA, 
                              d_blkcolptrB, d_blkrowidxB, blknB, 
                              d_blkrowptrC, d_blkrowidxC, d_blkcolidxC);
*/

    /* Calculating value of C */
    calculate_value_col_bin(d_blkrowptrA, d_blkcolidxA, NULL,
                            d_blkrowptrB, d_blkcolidxB, NULL,
                            d_blkrowptrC, d_blkrowidxC, d_blkcolidxC, NULL,
                            &bin,
                            blkmA, blkmB);

 //   cudaDeviceSynchronize();
 //   gettimeofday(&t2, NULL);

  //  double time_kernel = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
  //  printf("CUDA step2 new kernel = %.2f ms\n", time_kernel);



    //cudaFree(d_blkrowptrA);
    //cudaFree(d_blkcolidxA);
  //  cudaFree(d_blkrowptrB);
  //  cudaFree(d_blkcolidxB);
  //  cudaFree(d_blkrowptrC);
  //  cudaFree(d_blkrowptrC_offset);
  //  free(h_blkcolidxC);

    release_bin(bin);
 //   printf("step 2 success!\n");

    unsigned char *d_blkcsr_Ptr_C;
    cudaMalloc((void **)&d_blkcsr_Ptr_C, numblkC * BLOCK_SIZE * sizeof(unsigned char));
    //cudaMemset(d_blkcsr_Ptr_C, 0, numblkC * BLOCK_SIZE * sizeof(unsigned char));

    //void step3_cuda (int *blkrowptrA, int *blkcolidxA, int blkmA, int blknA, int numblkA, int nnzA, int *nnzb_A ,int mA,
    //        MAT_VAL_TYPE *blkcsr_Val_A , unsigned char  *blkcsr_Col_A , unsigned char *blkcsr_Ptr_A ,
    //        int *blkcolptrB, int *blkrowidxB, int blkmB, int blknB , int numblkB, int nnzB, int *nnzb_B ,
    //        MAT_VAL_TYPE *blkcsr_Val_B , unsigned char  *blkcsr_Col_B , unsigned char *blkcsr_Ptr_B ,
    //        int *blkrowptrC, int *blkcolidxC,int *d_blkrowidxC, int *d_blkcolidxC, unsigned char *d_blkcsr_Ptr_C,
    //        int *nnzb_C_golden,  int numblkC, int *nnzC)

    //int *d_blkrowptrC;
    //int *d_blkrowidxC;
    //int *d_blkcolidxC;
    int *d_nnzb_C;

    //cudaMalloc((void **)&d_blkrowptrC, (blkmA+1) * sizeof(int));
    //cudaMalloc((void **)&d_blkrowidxC, numblkC * sizeof(int));
    //cudaMalloc((void **)&d_blkcolidxC, numblkC * sizeof(int));
    cudaMalloc((void **)&d_nnzb_C, (numblkC+1) * sizeof(int));

    //cudaMemcpy(d_blkrowptrC,     blkrowptrC,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemcpy(d_blkrowidxC,     blkrowidxC,     numblkC * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemcpy(d_blkcolidxC,     blkcolidxC,     numblkC * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemset(d_nnzb_C, 0, (numblkC+1) * sizeof(int));

    int *d_blkid_smem;
    int *d_blkid_dns;

    //cudaMalloc((void **)&d_blkid_smem, numblkC * sizeof(int));
    //cudaMalloc((void **)&d_blkid_dns, numblkC * sizeof(int));

    int *d_blksmem_cnt;
    int *d_blkdns_cnt;

    //cudaMalloc((void **)&d_blksmem_cnt, 1 * sizeof(int));
    //cudaMalloc((void **)&d_blkdns_cnt, 1 * sizeof(int));

    //cudaMemset(d_blksmem_cnt, 0, 1 * sizeof(int));
    //cudaMemset(d_blkdns_cnt, 0, 1 * sizeof(int));

  //  struct timeval t1, t2;

  //  cudaDeviceSynchronize();
 //   gettimeofday(&t1, NULL);

    int num_threads = WARP_SIZE * WARP_PER_BLOCK;
    int num_blocks = ceil((double)numblkC/(double)WARP_PER_BLOCK);
    stir_spgemm_step3_cuda_kernel<128><<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A, d_blkmaskA,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B, d_blkmaskB,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, d_nnzb_C, d_blkmaskC, 
                              d_blksmem_cnt, d_blkdns_cnt, d_blkid_smem, d_blkid_dns, numblkC);

exclusive_scan_device_cuda(d_nnzb_C, numblkC+1);
int nnzC = 0;
cudaMemcpy(&nnzC,     &d_nnzb_C[numblkC],     sizeof(int), cudaMemcpyDeviceToHost);
*nnzC_computed = nnzC;
//*numblkC = nbc;

  //  cudaDeviceSynchronize();
  //  gettimeofday(&t2, NULL);

  //  double time_kernel = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
 //   printf("CUDA step3 kernel = %.2f ms, nnzC = %i\n", time_kernel, nnzC);


    //printf("move %i\n", 0x1 << 2);  //0001 -> 0010

    //cudaFree(d_blkrowptrA);
    //cudaFree(d_blkcolidxA); 
 //   cudaFree(d_nnzb_A); 
 //   cudaFree(d_blkcsr_Val_A);
  //  cudaFree(d_blkcsr_Col_A);
  //  cudaFree(d_blkcsr_Ptr_A);

  //  cudaFree(d_blkcolptrB);
   // cudaFree(d_blkrowidxB); 
   // cudaFree(d_nnzb_B); 
  //  cudaFree(d_blkcsr_Val_B);
  //  cudaFree(d_blkcsr_Col_B);
  //  cudaFree(d_blkcsr_Ptr_B);

    //cudaFree(d_blkrowptrC);
    //cudaFree(d_blkrowidxC);
    //cudaFree(d_blkcolidxC);
   // cudaFree(d_nnzb_C);
  //  free(h_nnzb_C);

  //  printf("step 3 success!\n");

unsigned char *d_blkcsr_Col_C;
    cudaMalloc((void **)&d_blkcsr_Col_C, nnzC * sizeof(unsigned char));
MAT_VAL_TYPE *d_blkcsr_Val_C;
    cudaMalloc((void **)&d_blkcsr_Val_C, nnzC * sizeof(MAT_VAL_TYPE));

    //void step4_cuda (int *blkrowptrA, int *blkcolidxA, int blkmA, int blknA, int numblkA, int nnzA, int *nnzb_A ,int mA,
    //        MAT_VAL_TYPE *blkcsr_Val_A , unsigned char  *blkcsr_Col_A , unsigned char *blkcsr_Ptr_A ,
    //        int *blkcolptrB, int *blkrowidxB, int blkmB, int blknB , int numblkB, int nnzB, int *nnzb_B ,
    //        MAT_VAL_TYPE *blkcsr_Val_B , unsigned char  *blkcsr_Col_B , unsigned char *blkcsr_Ptr_B ,
    //        int *blkrowptrC, int *blkcolidxC,
    //        int *d_blkrowidxC, int *d_blkcolidxC, 
    //        unsigned char *d_blkcsr_Ptr_C, unsigned char *d_blkcsr_Col_C, MAT_VAL_TYPE *d_blkcsr_Val_C,
    //        int *nnzb_C,  int numblkC, 
    //        MAT_VAL_TYPE *blkcsr_Val_C_golden, 
    //        unsigned char *blkcsr_Col_C_golden, 
    //        unsigned char *blkcsr_Ptr_C_golden, int nnzC)


    //int *d_blkrowptrA;
    //int *d_blkcolidxA; 
  //  int *d_nnzb_A; 
  //  MAT_VAL_TYPE *d_blkcsr_Val_A;
  //  unsigned char *d_blkcsr_Col_A;
  //  unsigned char *d_blkcsr_Ptr_A;

    //cudaMalloc((void **)&d_blkrowptrA, (blkmA+1) * sizeof(int));
    //cudaMalloc((void **)&d_blkcolidxA, numblkA * sizeof(int));
  //  cudaMalloc((void **)&d_nnzb_A, numblkA * sizeof(int));
  //  cudaMalloc((void **)&d_blkcsr_Val_A, nnzA * sizeof(MAT_VAL_TYPE));
  //  cudaMalloc((void **)&d_blkcsr_Col_A, nnzA * sizeof(unsigned char));
  //  cudaMalloc((void **)&d_blkcsr_Ptr_A, numblkA * BLOCK_SIZE  * sizeof(unsigned char));

    //cudaMemcpy(d_blkrowptrA,     blkrowptrA,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemcpy(d_blkcolidxA,     blkcolidxA,     numblkA * sizeof(int),              cudaMemcpyHostToDevice);
 //   cudaMemcpy(d_nnzb_A,     nnzb_A,     numblkA * sizeof(int),              cudaMemcpyHostToDevice);
 //   cudaMemcpy(d_blkcsr_Val_A,     blkcsr_Val_A,     nnzA * sizeof(MAT_VAL_TYPE),              cudaMemcpyHostToDevice);
 //   cudaMemcpy(d_blkcsr_Col_A,     blkcsr_Col_A,     nnzA * sizeof(unsigned char),              cudaMemcpyHostToDevice);
 //   cudaMemcpy(d_blkcsr_Ptr_A,     blkcsr_Ptr_A,     numblkA * BLOCK_SIZE * sizeof(unsigned char),              cudaMemcpyHostToDevice);

  //  int *d_blkcolptrB;
   // int *d_blkrowidxB; 
  ///  int *d_nnzb_B; 
  //  MAT_VAL_TYPE *d_blkcsr_Val_B;
  //  unsigned char *d_blkcsr_Col_B;
 //   unsigned char *d_blkcsr_Ptr_B;

   // cudaMalloc((void **)&d_blkcolptrB, (blknB+1) * sizeof(int));
   // cudaMalloc((void **)&d_blkrowidxB, numblkB * sizeof(int));
   /// cudaMalloc((void **)&d_nnzb_B, numblkB * sizeof(int));
  //  cudaMalloc((void **)&d_blkcsr_Val_B, nnzB * sizeof(MAT_VAL_TYPE));
   // cudaMalloc((void **)&d_blkcsr_Col_B, nnzB * sizeof(unsigned char));
   // cudaMalloc((void **)&d_blkcsr_Ptr_B, numblkB * BLOCK_SIZE  * sizeof(unsigned char));

  //  cudaMemcpy(d_blkcolptrB,     blkcolptrB,     (blknB+1) * sizeof(int),              cudaMemcpyHostToDevice);
  //  cudaMemcpy(d_blkrowidxB,     blkrowidxB,     numblkB * sizeof(int),              cudaMemcpyHostToDevice);
  //  cudaMemcpy(d_nnzb_B,     nnzb_B,     numblkB * sizeof(int),              cudaMemcpyHostToDevice);
 //   cudaMemcpy(d_blkcsr_Val_B,     blkcsr_Val_B,     nnzB * sizeof(MAT_VAL_TYPE),              cudaMemcpyHostToDevice);
  //  cudaMemcpy(d_blkcsr_Col_B,     blkcsr_Col_B,     nnzB * sizeof(unsigned char),              cudaMemcpyHostToDevice);
 //   cudaMemcpy(d_blkcsr_Ptr_B,     blkcsr_Ptr_B,     numblkB * BLOCK_SIZE * sizeof(unsigned char),              cudaMemcpyHostToDevice);


    //int *d_blkrowptrC;
    //int *d_blkrowidxC;
    //int *d_blkcolidxC;
   // int *d_nnzb_C;

    //cudaMalloc((void **)&d_blkrowptrC, (blkmA+1) * sizeof(int));
    //cudaMalloc((void **)&d_blkrowidxC, numblkC * sizeof(int));
    //cudaMalloc((void **)&d_blkcolidxC, numblkC * sizeof(int));
  //  cudaMalloc((void **)&d_nnzb_C, (numblkC+1) * sizeof(int));

    //cudaMemcpy(d_blkrowptrC,     blkrowptrC,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemcpy(d_blkrowidxC,     blkrowidxC,     numblkC * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemcpy(d_blkcolidxC,     blkcolidxC,     numblkC * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemset(d_nnzb_C, 0, numblkC * sizeof(int));
//cudaMemcpy(d_nnzb_C,     nnzb_C,     (numblkC+1) * sizeof(int),              cudaMemcpyHostToDevice);

  //  struct timeval t1, t2;

   // cudaDeviceSynchronize();
   // gettimeofday(&t1, NULL);
/*
int blksmem_cnt = 0;
int blkdns_cnt = 0;
cudaMemcpy(&blksmem_cnt,     d_blksmem_cnt,     sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(&blkdns_cnt,     d_blkdns_cnt,     sizeof(int), cudaMemcpyDeviceToHost);

     num_threads = WARP_SIZE * WARP_PER_BLOCK;
     num_blocks = ceil((double)blksmem_cnt/(double)WARP_PER_BLOCK);
    stir_spgemm_step4_cuda_kernel_smem<32><<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, 
d_blkcsr_Col_C,d_blkcsr_Val_C,
d_nnzb_C, d_blkmaskC, blksmem_cnt, d_blkid_smem);

    num_blocks = ceil((double)blkdns_cnt/(double)WARP_PER_BLOCK);
    stir_spgemm_step4_cuda_kernel_dns<<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, 
d_blkcsr_Col_C,d_blkcsr_Val_C,
d_nnzb_C, blkdns_cnt, d_blkid_dns);
*/





     num_threads = WARP_SIZE * WARP_PER_BLOCK;
     num_blocks = ceil((double)numblkC/(double)WARP_PER_BLOCK);
    stir_spgemm_step4_cuda_kernel_dns<64><<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, 
d_blkcsr_Col_C,d_blkcsr_Val_C,
d_nnzb_C, d_blkmaskC, numblkC, NULL);





/*

     num_threads = WARP_SIZE * WARP_PER_BLOCK;
     num_blocks = ceil((double)numblkC/(double)WARP_PER_BLOCK);
    stir_spgemm_step4_cuda_kernel_smem<256><<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, 
d_blkcsr_Col_C,d_blkcsr_Val_C, 
d_nnzb_C, d_blkmaskC, numblkC);
*/
/*
    stir_spgemm_step4_cuda_kernel_dns<<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, 
d_blkcsr_Col_C,d_blkcsr_Val_C, 
d_nnzb_C, d_blkmaskC, numblkC);
*/

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
stir_spgemm_time += time;

    printf("CUDA stepall kernel = %.2f ms\n", time);
printf("CUDA numblkC = %i\n", numblkC);
    printf("CUDA nnzC = %i\n", nnzC);

if (ri != REPEAT_NUM - 1)
{
    cudaFree(d_blkrowptrC);
    cudaFree(d_blkrowidxC);
    cudaFree(d_blkcolidxC);
    cudaFree(d_nnzb_C);
    cudaFree(d_blkcsr_Ptr_C);
    cudaFree(d_blkcsr_Col_C);
    cudaFree(d_blkcsr_Val_C);
}
else if(result_check == 1 && ri == REPEAT_NUM - 1)
{

    int *h_blkrowptrC = (int *)malloc((blkmA+1)*sizeof(int));
    cudaMemcpy(h_blkrowptrC,     d_blkrowptrC,     (blkmA+1) * sizeof(int), cudaMemcpyDeviceToHost);

    int errcnt = 0;
    for (int i = 0; i < (blkmA+1); i++)
        if (h_blkrowptrC[i] != blkrowptrC_golden[i]){ //printf("%i, %i\n", h_blkrowptrC[i], blkrowptrC_golden[i]);
            errcnt++;}
    printf("step 1 new, blkrowptrC, errcnt = %i\n", errcnt);

    int *h_blkcolidxC = (int *)malloc(numblkC*sizeof(int));
    cudaMemcpy(h_blkcolidxC,     d_blkcolidxC,     numblkC * sizeof(int), cudaMemcpyDeviceToHost);
/*
    for(int blki =0;blki < blkmA ;blki ++)
    {
        quick_sort_key(h_blkcolidxC + blkrowptrC[blki],blkrowptrC[blki+1] - blkrowptrC[blki]);
    }

    cudaMemcpy(d_blkcolidxC,     h_blkcolidxC,     numblkC * sizeof(int), cudaMemcpyHostToDevice);
*/
     errcnt = 0;
    for (int i = 0; i < numblkC; i++)
       if (h_blkcolidxC[i] != blkcolidxC_golden[i])
            errcnt++;
    printf("step 2 new, h_blkcolidxC, errcnt = %i\n", errcnt);

    int *h_nnzb_C = (int *)malloc((numblkC+1)*sizeof(int));
    cudaMemcpy(h_nnzb_C,     d_nnzb_C,     (numblkC+1) * sizeof(int), cudaMemcpyDeviceToHost);

     errcnt = 0;
    for (int i = 0; i < (numblkC+1); i++)
        if (h_nnzb_C[i] != nnzb_C_golden[i])
            {//printf("[%i] %i, %i\n", i, h_nnzb_C[i], nnzb_C_golden[i]); 
          errcnt++;}
    printf("step 3, h_nnzb_C, errcnt = %i\n", errcnt);


    MAT_VAL_TYPE *h_blkcsr_Val_C = (MAT_VAL_TYPE *)malloc(nnzC*sizeof(MAT_VAL_TYPE));
    cudaMemcpy(h_blkcsr_Val_C,     d_blkcsr_Val_C,     nnzC*sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);

    unsigned char *h_blkcsr_Col_C = (unsigned char *)malloc(nnzC*sizeof(unsigned char));
    cudaMemcpy(h_blkcsr_Col_C,     d_blkcsr_Col_C,     nnzC*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    unsigned char *h_blkcsr_Ptr_C = (unsigned char *)malloc(BLOCK_SIZE*numblkC*sizeof(unsigned char));
    cudaMemcpy(h_blkcsr_Ptr_C,     d_blkcsr_Ptr_C,     BLOCK_SIZE*numblkC*sizeof(unsigned char), cudaMemcpyDeviceToHost);

     errcnt = 0;
    for (int i = 0; i < BLOCK_SIZE*numblkC; i++)
        if (h_blkcsr_Ptr_C[i] != blkcsr_Ptr_C_golden[i])
            {errcnt++;}
    printf("step 4, h_blkcsr_Ptr_C, errcnt = %i\n", errcnt);

    errcnt = 0;
    for (int i = 0; i < nnzC; i++)
        if (h_blkcsr_Col_C[i] != blkcsr_Col_C_golden[i])
            {errcnt++;}
    printf("step 4, h_blkcsr_Col_C, errcnt = %i\n", errcnt);

    errcnt = 0;
    for (int i = 0; i < nnzC; i++)
        if (h_blkcsr_Val_C[i] != blkcsr_Val_C_golden[i])
            {//printf("%f, %f\n", h_blkcsr_Val_C[i], blkcsr_Val_C_golden[i]); 
            errcnt++;}
    printf("step 4, h_blkcsr_Val_C, errcnt = %i\n", errcnt);

free(h_blkrowptrC);
free(h_blkcolidxC);
free(h_nnzb_C);
free(h_blkcsr_Val_C);
free(h_blkcsr_Col_C);
free(h_blkcsr_Ptr_C);

    cudaFree(d_blkrowptrC);
    cudaFree(d_blkrowidxC);
    cudaFree(d_blkcolidxC);
    cudaFree(d_nnzb_C);
    cudaFree(d_blkcsr_Ptr_C);
    cudaFree(d_blkcsr_Col_C);
    cudaFree(h_blkcsr_Val_C);
    cudaFree(d_blkmaskC);
}

}


*compression_rate = (double)nnzCub / (double)(*nnzC_computed);
stir_spgemm_time /= REPEAT_NUM;
*time_stir = stir_spgemm_time;
*gflops_stir = 2.0 * (double)nnzCub / (stir_spgemm_time * 1e6);

printf("stir_spgemm_time = %4.2f, gflops = %4.2f\n", stir_spgemm_time);


    cudaFree(d_blkrowptrA);
    cudaFree(d_blkcolidxA); 
    cudaFree(d_nnzb_A); 
    cudaFree(d_blkcsr_Val_A);
    cudaFree(d_blkcsr_Col_A);
    cudaFree(d_blkcsr_Ptr_A);

    cudaFree(d_blkcolptrB);
    cudaFree(d_blkrowidxB); 
    cudaFree(d_blkrowptrB);
    cudaFree(d_blkcolidxB);
    cudaFree(d_nnzb_B); 
    cudaFree(d_blkcsr_Val_B);
    cudaFree(d_blkcsr_Col_B);
    cudaFree(d_blkcsr_Ptr_B);




   // printf("step 4 success!\n");
}

int main(int argc, char ** argv)
{

	if (argc < 4)
    {
        printf("Run the code by './test -d 0 matrix.mtx'.\n");
        return 0;
    }
	
    printf("--------------------------------!!!!!!!!------------------------------------\n");
    
        int device_id = 0;
    bool check_result = 0;

    // "Usage: ``./spgemm -d 0 -check 0 A.mtx B.mtx'' for AB=C on device 0, no check"
    int argi = 1;

    // load device id
    char *devstr;
    if(argc > argi)
    {
        devstr = argv[argi];
        argi++;
    }

    if (strcmp(devstr, "-d") != 0) return 0;

    if(argc > argi)
    {
        device_id = atoi(argv[argi]);
        argi++;
    }
    printf("device_id = %i\n", device_id);
    
        // set device
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    printf("---------------------------------------------------------------\n");
    printf("Device [ %i ] %s @ %4.2f MHz\n",
           device_id, deviceProp.name, deviceProp.clockRate * 1e-3f);
           

 	struct timeval t1, t2;
	int rowA;
	int colA;
	MAT_PTR_TYPE nnzA;
	int isSymmetricA;
	SMatrix matrixA;


	char  *filename;
    filename = argv[3];
    printf("MAT: -------------- %s --------------\n", filename);

    // load mtx A data to the csr format
    gettimeofday(&t1, NULL);
    mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA, &matrixA.rowpointer, &matrixA.columnindex, &matrixA.value, filename);
    gettimeofday(&t2, NULL);
    double time_loadmat  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("input matrix A: ( %i, %i ) nnz = %i\n loadfile time    = %4.5f sec\n", rowA, colA, nnzA, time_loadmat/1000.0);

if (!AAT && rowA != colA)
{
printf("matrix squaring must have rowA == colA. Exit.\n");
    return 0;
}


    printf("the blocksize = %d\n",BLOCK_SIZE);
	for (int i = 0; i < nnzA; i++)
	    matrixA.value[i] = i % 10;
/*
    if (rowA != colA)
    {
        printf("This code only computes square matrices.\n Exit.\n");
        return 0;
    }
*/
  	SMatrix matrixB;
int rowB, colB, nnzB;
MAT_PTR_TYPE *cscColPtrA;
int *cscRowIdxA;
MAT_VAL_TYPE *cscValA ;

	// rowB=colA;
	// colB=rowA;
    // nnzB = nnzA ;

if (AAT)
{
	 rowB=colA;
	 colB=rowA;
     nnzB = nnzA ;

    cscColPtrA = (MAT_PTR_TYPE *)malloc((colA+1) * sizeof(MAT_PTR_TYPE));
    cscRowIdxA = (int *)malloc(nnzA   * sizeof(int));
    cscValA    = (MAT_VAL_TYPE *)malloc(nnzA  * sizeof(MAT_VAL_TYPE));

	 // transpose A from csr to csc
    matrix_transposition(rowA, colA, nnzA, matrixA.rowpointer, matrixA.columnindex, matrixA.value,cscRowIdxA, cscColPtrA, cscValA);

	matrixB.rowpointer = cscColPtrA;
    matrixB.columnindex = cscRowIdxA;
    matrixB.value    = cscValA;
}
else
{
	 rowB=rowA;
	 colB=colA;
     nnzB = nnzA ;

	matrixB.rowpointer = matrixA.rowpointer;
    matrixB.columnindex = matrixA.columnindex;
    matrixB.value    = matrixA.value;
}

    // calculate bytes and flops consumed
    unsigned long long int nnzCub = 0;
    for (int i = 0; i < nnzA; i++)
    {
        int rowidx = matrixA.columnindex[i];
        nnzCub += matrixB.rowpointer[rowidx + 1] - matrixB.rowpointer[rowidx];
    }

    printf("SpGEMM nnzCub = %lld\n", nnzCub);

    if (BLOCK_SIZE>rowA){
		printf("Error!\n");
		return 0;
	}





    int rbnum_A =0;
    int cbnum_A =0;
  
//    memset(rowblock_ptr_A,0,(rbnum_A+1)*sizeof(MAT_PTR_TYPE));
    //int *blkcolIdx_A ;
    int blknnz_A =0;
    int ptrA_length =0 ;

    blocknnz(&rbnum_A,&cbnum_A,matrixA.columnindex,matrixA.rowpointer,rowA,colA,&blknnz_A);

    MAT_PTR_TYPE *rowblock_ptr_A = (MAT_PTR_TYPE *)malloc((rbnum_A+1)*sizeof(MAT_PTR_TYPE)); 
    int *blkcolIdx_A = (int *)malloc (blknnz_A * sizeof(int ) );
    blkmessage(&rbnum_A,&cbnum_A,matrixA.columnindex,matrixA.rowpointer,rowA,colA,rowblock_ptr_A, blkcolIdx_A);

    for(int blki =0;blki < rbnum_A ;blki ++)
    {
        quick_sort_key(blkcolIdx_A + rowblock_ptr_A[blki],rowblock_ptr_A[blki+1] - rowblock_ptr_A[blki]);
    }

/*    for (int i=0;i<rowblock_ptr_A[rbnum_A];i++)
    {
        printf("%d   ",blkcolIdx_A[i]);
    }
    printf("\n") ;
     printf("\n") ;
*/


    int csrrbnum_B =0;
    int csrcbnum_B =0;
    int csrblknnz_B =0;
    //int ptrB_length =0 ;

    blocknnz(&csrrbnum_B,&csrcbnum_B,matrixB.columnindex,matrixB.rowpointer,rowB,colB,&csrblknnz_B);

    MAT_PTR_TYPE *rowblock_ptr_B = (MAT_PTR_TYPE *)malloc((csrrbnum_B+1)*sizeof(MAT_PTR_TYPE)); 
    int *blkcolIdx_B = (int *)malloc (csrblknnz_B * sizeof(int ) );
    blkmessage(&csrrbnum_B,&csrcbnum_B,matrixB.columnindex,matrixB.rowpointer,rowB,colB,rowblock_ptr_B, blkcolIdx_B);

    for(int blki =0;blki < csrrbnum_B ;blki ++)
    {
        quick_sort_key(blkcolIdx_B + rowblock_ptr_B[blki],rowblock_ptr_B[blki+1] - rowblock_ptr_B[blki]);
    }


/*    for (int i=0;i<rbnum_A+1;i++)
	{
		printf("%d    ",rowblock_ptr_A[i]);
	}
    printf("\n");
    printf("blknnz_A =%d\n",blknnz_A) ;

	for (int i=0;i<blknnz_A;i++)
	{
		printf("%d    ",blkcolIdx_A[i]);
	}
    printf("\n");
*/
    MAT_PTR_TYPE *cscColPtrB = (MAT_PTR_TYPE *)malloc((colB+1) * sizeof(MAT_PTR_TYPE));
    int *cscRowIdxB = (int *)malloc(nnzB   * sizeof(int));
    MAT_VAL_TYPE *cscValB    = (MAT_VAL_TYPE *)malloc(nnzB  * sizeof(MAT_VAL_TYPE));

    // transpose B from csr to csc
    matrix_transposition(rowB, colB, nnzB, matrixB.rowpointer, matrixB.columnindex, matrixB.value,cscRowIdxB, cscColPtrB, cscValB);
    


    int rbnum_B_trans =0;
    int cbnum_B_trans =0;

   
//    memset(rowblock_ptr_A,0,(rbnum_A+1)*sizeof(MAT_PTR_TYPE));
    int *blkcolIdx_B_trans ;
    int blknnz_B =0;
    blocknnz(&rbnum_B_trans,&cbnum_B_trans,cscRowIdxB,cscColPtrB,colB,rowB,&blknnz_B);

    MAT_PTR_TYPE *rowblock_ptr_B_trans = (MAT_PTR_TYPE *)malloc((rbnum_B_trans+1)*sizeof(MAT_PTR_TYPE)); 

    blkcolIdx_B_trans = (int *)malloc (blknnz_B * sizeof(int ) );
    blkmessage(&rbnum_B_trans,&cbnum_B_trans,cscRowIdxB,cscColPtrB,colB,rowB,rowblock_ptr_B_trans, blkcolIdx_B_trans);

    

    int rbnum_B = cbnum_B_trans ;
    int cbnum_B = rbnum_B_trans ;

    MAT_PTR_TYPE *colblock_ptr_B = (MAT_PTR_TYPE *)malloc((cbnum_B+1)*sizeof(MAT_PTR_TYPE)); 
    int *blkrowIdx_B = (int *)malloc (blknnz_B * sizeof(int ) ) ;

    colblock_ptr_B = rowblock_ptr_B_trans ;
    blkrowIdx_B = blkcolIdx_B_trans ;

    for(int blki =0;blki < cbnum_B ;blki ++)
    {
        quick_sort_key(blkrowIdx_B + colblock_ptr_B[blki],colblock_ptr_B[blki+1] - colblock_ptr_B[blki]);
    }

    MAT_PTR_TYPE *rowblock_ptr_C = (MAT_PTR_TYPE *)malloc((rbnum_A+1)*sizeof(MAT_PTR_TYPE)); 
    int blknnz_C =0;

    int numblkA = rowblock_ptr_A[rbnum_A];
    int numblkB = colblock_ptr_B[cbnum_B];
    int numblkC = 0;
    step1 (rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,
          colblock_ptr_B,blkrowIdx_B,rbnum_B,cbnum_B,
          rowblock_ptr_C, &numblkC);

    //step1_cuda (rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,numblkA,
    //      colblock_ptr_B,blkrowIdx_B,rbnum_B,cbnum_B,numblkB,
    //      rowblock_ptr_C, &numblkC);

sfBIN bin;
    step1_cuda_new (&bin, rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,numblkA,
          rowblock_ptr_B,blkcolIdx_B,csrrbnum_B,csrcbnum_B,numblkB,
          rowblock_ptr_C, &numblkC);



//   int *d_rowblock_ptr_C;
//cudaMalloc((void **)&d_rowblock_ptr_C, (rbnum_A+1) * sizeof(int));
//   cudaMemcpy(d_rowblock_ptr_C,     rowblock_ptr_C,     (rbnum_A+1) * sizeof(int),              cudaMemcpyHostToDevice);

    
 /*   for (int i=0;i<rbnum_A+1 ;i++)
    {
        printf("%d   ",rowblock_ptr_C[i]);
    }
    printf("\n");
*/
    printf("step 1 success!\n");
    //int *blkcolIdx_C = (int *)malloc(rowblock_ptr_C[rbnum_A] * sizeof (int)) ;
    int *blkcolIdx_C = (int *)malloc(numblkC * sizeof (int)) ;
    printf("rowblock_ptr_C[rbnum_A] = %i\n", rowblock_ptr_C[rbnum_A]);

    int *d_blkcolIdx_C;
    cudaMalloc((void **)&d_blkcolIdx_C, numblkC * sizeof(int));
    int *d_blkrowIdx_C;
    cudaMalloc((void **)&d_blkrowIdx_C, numblkC * sizeof(int));
    //cudaMemcpy(d_rowblock_ptr_C,     rowblock_ptr_C,     (rbnum_A+1) * sizeof(int),

    step2 (rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,
          colblock_ptr_B,blkrowIdx_B,rbnum_B,cbnum_B,
          rowblock_ptr_C,blkcolIdx_C);

    //step2_cuda (rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,numblkA,
    //      colblock_ptr_B,blkrowIdx_B,rbnum_B,cbnum_B,numblkB,
    //      rowblock_ptr_C,blkcolIdx_C, d_blkrowIdx_C, d_blkcolIdx_C,numblkC);

    step2_cuda_new (&bin, rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,numblkA,
          rowblock_ptr_B,blkcolIdx_B,csrrbnum_B,csrcbnum_B,numblkB,
          rowblock_ptr_C,blkcolIdx_C, d_blkrowIdx_C, d_blkcolIdx_C,numblkC);

    release_bin(bin);

    printf("step 2 success!\n");
/*for (int i=0;i<rowblock_ptr_C[rbnum_A];i++)
{
    printf("%d     ",blkcolIdx_C[i]);
}
printf("\n");
*/
    int ptrlen_A =rowblock_ptr_A[rbnum_A] * BLOCK_SIZE ;
    int *nnzb_A = (int *)malloc((rowblock_ptr_A[rbnum_A] + 1) * sizeof (int)) ;

    int ptrlen_B = colblock_ptr_B[cbnum_B] * BLOCK_SIZE ;
    int *nnzb_B = (int *)malloc((colblock_ptr_B[cbnum_B] +1) * sizeof (int)) ;
     
    for (int blki=0;blki<rbnum_A;blki++)
	{
        int rowbnum=rowblock_ptr_A[blki+1]-rowblock_ptr_A[blki];
        int *rownnzA=(int *)malloc((rowbnum) * sizeof(int));
        memset(rownnzA,0,rowbnum *sizeof(int));
        int rowlength= blki==rbnum_A-1 ? rowA-(rbnum_A-1)*BLOCK_SIZE : BLOCK_SIZE ;
     //   printf("rowlength=%d\n",rowlength);
        int start= blki*BLOCK_SIZE;
        int end = blki==rbnum_A-1 ?  rowA : (blki+1)*BLOCK_SIZE ;
        for (int j=matrixA.rowpointer[start];j<matrixA.rowpointer[end];j++)
        {
            int ki;
            for (int k=rowblock_ptr_A[blki],ki=0;k<rowblock_ptr_A[blki+1],ki<rowbnum;k++,ki++)
            {
                int kcstart=blkcolIdx_A[k]*BLOCK_SIZE;
                int kcend= blkcolIdx_A[k]== (cbnum_A-1) ?  colA: (blkcolIdx_A[k]+1)*BLOCK_SIZE;
                if (matrixA.columnindex[j]>=kcstart&&matrixA.columnindex[j]<kcend)
                {
                    rownnzA[ki]++;
                    break;
                }
		    }
	    }
        for(int bi=0;bi<rowbnum;bi++)
        {
            nnzb_A[rowblock_ptr_A[blki]+bi]=rownnzA[bi];
        }
    //    ptrlen_A += rowbnum * rowlength ;
        free(rownnzA);
    }
    exclusive_scan (nnzb_A,rowblock_ptr_A[rbnum_A] + 1) ;

    for (int blki=0;blki<cbnum_B;blki++)
	{
        int rowbnum_B_trans=colblock_ptr_B[blki+1]-colblock_ptr_B[blki];
        int *rownnzB_trans=(int *)malloc((rowbnum_B_trans) * sizeof(int));
        memset(rownnzB_trans,0,rowbnum_B_trans *sizeof(int));
        //int rowlength= blki==cbnum_B-1 ? colB-(colB-1)*BLOCK_SIZE : BLOCK_SIZE ;
     //   printf("rowlength=%d\n",rowlength);
        int start= blki*BLOCK_SIZE;
        int end = blki==cbnum_B-1 ?  colB : (blki+1)*BLOCK_SIZE ;
        for (int j=cscColPtrB[start];j<cscColPtrB[end];j++)
        //for (int j=matrixA.rowpointer[start];j<matrixA.rowpointer[end];j++)
        {
            int ki;
            for (int k=colblock_ptr_B[blki],ki=0;k<colblock_ptr_B[blki+1],ki<rowbnum_B_trans;k++,ki++)
            {
                //int kcstart=blkcolIdx_B_trans[k]*BLOCK_SIZE;
                int kcstart=blkrowIdx_B[k]*BLOCK_SIZE;
                //int kcend= blkcolIdx_B_trans[k]== (rbnum_B-1) ?  rowB: (blkcolIdx_B_trans[k]+1)*BLOCK_SIZE;
                int kcend= blkrowIdx_B[k]== (rbnum_B-1) ?  rowB: (blkrowIdx_B[k]+1)*BLOCK_SIZE;
                if (cscRowIdxB[j]>=kcstart&&cscRowIdxB[j]<kcend)
                //if (matrixA.columnindex[j]>=kcstart&&matrixA.columnindex[j]<kcend)
                {
                    rownnzB_trans[ki]++;
                    break;
                }
		    }
	    }
        for(int bi=0;bi<rowbnum_B_trans;bi++)
        {
         //   int collength_trans = blkcolIdx_B_trans[rowblock_ptr_B_trans[blki]+bi] == rbnum_B -1 ? rowB - (rbnum_B -1) * BLOCK_SIZE : BLOCK_SIZE ;
        //    printf("collength_trans = %d\n",collength_trans);
            nnzb_B[colblock_ptr_B[blki]+bi]=rownnzB_trans[bi];
         //   ptrlen_B += collength_trans ;
        }
        free(rownnzB_trans);
    }
    printf("ptrlen_B = %d\n",ptrlen_B);
    exclusive_scan (nnzb_B,colblock_ptr_B[cbnum_B] + 1) ;





    MAT_VAL_TYPE *blkcsr_Val_A=(MAT_VAL_TYPE*)malloc((nnzA)*sizeof(MAT_VAL_TYPE));
    unsigned char  *blk_idx_A=(unsigned char*)malloc((nnzA)*sizeof(unsigned char));
    unsigned char  *blkcsr_Col_A=(unsigned char*)malloc((nnzA)*sizeof(unsigned char));
    unsigned char *blkcsr_Ptr_A=(unsigned char*)malloc((ptrlen_A)*sizeof(unsigned char));
  
 //   int csrvid =0;
 //   int csrpid =0;


    MAT_VAL_TYPE *blkcsr_Val_B=(MAT_VAL_TYPE*)malloc((nnzB)*sizeof(MAT_VAL_TYPE));
    unsigned char  *blkcsr_Col_B=(unsigned char*)malloc((nnzB)*sizeof(unsigned char));
    unsigned char *blkcsr_Ptr_B=(unsigned char*)malloc((ptrlen_B)*sizeof(unsigned char));
    int ptroffset_B =0;
   

    

    // block array of A
unsigned short  *blkmaskA = (unsigned short *)malloc(numblkA * BLOCK_SIZE * sizeof(unsigned short));
    memset(blkmaskA,0,numblkA * BLOCK_SIZE * sizeof(unsigned short)) ;

    for (int blki=0;blki<rbnum_A;blki++)
	{
        int rowbnum=rowblock_ptr_A[blki+1]-rowblock_ptr_A[blki];
        SMatrix *subrowmatrixA=(SMatrix *)malloc(rowbnum*sizeof(SMatrix));
        int rowlength= blki==rbnum_A-1 ? rowA-(rbnum_A-1)*BLOCK_SIZE : BLOCK_SIZE ;
     //   printf("rowlength=%d\n",rowlength);
    
        for (int bi=0;bi<rowbnum;bi++) 
        {
           subrowmatrixA[bi].value=(MAT_VAL_TYPE*)malloc((nnzb_A[rowblock_ptr_A[blki]+bi +1]- nnzb_A[rowblock_ptr_A[blki]+bi ])*sizeof(MAT_VAL_TYPE));
           subrowmatrixA[bi].columnindex=(int *)malloc((nnzb_A[rowblock_ptr_A[blki]+bi +1]- nnzb_A[rowblock_ptr_A[blki]+bi ])*sizeof(int));
          
           subrowmatrixA[bi].rowpointer=(MAT_PTR_TYPE *)malloc((rowlength+1)*sizeof(MAT_PTR_TYPE));
           memset(subrowmatrixA[bi].rowpointer,0,(rowlength+1)*sizeof(MAT_PTR_TYPE));
        }

        int start= blki*BLOCK_SIZE;
        int end = blki==rbnum_A-1 ?  rowA : (blki+1)*BLOCK_SIZE ;
        int *num=(int*)malloc((rowbnum)*sizeof(int));
	    memset(num,0,(rowbnum)*sizeof(int));

        for (int ri=0;ri<rowlength;ri++)
        {
            for (int j=matrixA.rowpointer[start+ri];j<matrixA.rowpointer[start+ri+1];j++)
            {
                int ki;
                for (int k=rowblock_ptr_A[blki],ki=0;k<rowblock_ptr_A[blki+1],ki<rowbnum;k++,ki++)
                {
                    int kcstart=blkcolIdx_A[k]*BLOCK_SIZE;
                    int kcend= blkcolIdx_A[k]== (cbnum_A-1) ?  colA: (blkcolIdx_A[k]+1)*BLOCK_SIZE;
                    if (matrixA.columnindex[j]>=kcstart&&matrixA.columnindex[j]<kcend)
                    {
                        num[ki]++;
                        subrowmatrixA[ki].value[num[ki]-1]=matrixA.value[j];
                        subrowmatrixA[ki].columnindex[num[ki]-1]=matrixA.columnindex[j]-blkcolIdx_A[k]*BLOCK_SIZE;
                        break; 
                    }
                }
            }
            for (int bi=0;bi<rowbnum;bi++){
                    subrowmatrixA[bi].rowpointer[ri+1]=num[bi];
            }   
	    }
     
        for(int bi=0;bi<rowbnum;bi++)
        {
         /*   for (int kk=0;kk<blknnz[rowblock_ptr[blki]+bi + 1] - blknnz[rowblock_ptr[blki]+bi ]; kk++ )
            {
                printf("%d    ",(int)subrowmatrixA[bi].value[kk]);
            }
            printf("\n") ;
             printf("\n") ;
        */
            int collength = blkcolIdx_A[rowblock_ptr_A[blki]+bi] == cbnum_A-1 ? colA - (cbnum_A-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
            //CSR val&col

            for ( int ari = 0;ari <rowlength ;ari ++)
            {
                for (int k=subrowmatrixA[bi].rowpointer[ari] ;k < subrowmatrixA[bi].rowpointer[ari + 1];k++)
                {
int colidx = subrowmatrixA[bi].columnindex[k];
                    blkcsr_Val_A[nnzb_A[rowblock_ptr_A[blki]+bi ]+ k]=subrowmatrixA[bi].value[k] ;

                    blk_idx_A[nnzb_A[rowblock_ptr_A[blki]+bi ]+ k] = (ari << 4) + subrowmatrixA[bi].columnindex[k] ;
                    blkmaskA[(rowblock_ptr_A[blki] + bi) * BLOCK_SIZE + ari] |= (0x1 << (BLOCK_SIZE - colidx - 1));

                //    res[count] = (str1[i] << 4) + str2[i];
                    blkcsr_Col_A[nnzb_A[rowblock_ptr_A[blki]+bi ]+ k]=subrowmatrixA[bi].columnindex[k];

                }
                blkcsr_Ptr_A[(rowblock_ptr_A[blki]+ bi ) * BLOCK_SIZE + ari]=subrowmatrixA[bi].rowpointer[ari];

            }
/*
            for (int k=0;k<nnzb_A[rowblock_ptr_A[blki]+bi +1] -nnzb_A[rowblock_ptr_A[blki]+bi]; k++)
            {
                blkcsr_Val_A[nnzb_A[rowblock_ptr_A[blki]+bi ]+ k]=subrowmatrixA[bi].value[k] ;
        //    Block_Val [blknnz[rowblock_ptr[blki] + bi] + k] = subrowmatrixA[bi].value[k] ;
                blkcsr_Col_A[nnzb_A[rowblock_ptr_A[blki]+bi ]+ k]=subrowmatrixA[bi].columnindex[k];
             //   csrvid++;
            }
            //CSR ptr

            for (int jid=0;jid<rowlength;jid++)
            {
                blkcsr_Ptr_A[(rowblock_ptr_A[blki]+ bi ) * BLOCK_SIZE + jid]=subrowmatrixA[bi].rowpointer[jid];
            //    csrpid++;
            }
*/
            for (int pid =rowlength ; pid < BLOCK_SIZE;pid ++)
            {
                blkcsr_Ptr_A[(rowblock_ptr_A[blki]+ bi ) * BLOCK_SIZE + pid] = subrowmatrixA[bi].rowpointer[rowlength] ;
            }
        
        }
    
        for (int bi=0;bi<rowbnum;bi++)
        {
            free(subrowmatrixA[bi].value);
            free(subrowmatrixA[bi].columnindex);
            free(subrowmatrixA[bi].rowpointer);
        }
        free(subrowmatrixA);
        free(num);
    }

unsigned short  *blkmaskB = (unsigned short *)malloc(numblkB * BLOCK_SIZE * sizeof(unsigned short));
    memset(blkmaskB,0,numblkB * BLOCK_SIZE * sizeof(unsigned short)) ;

    for (int blki=0;blki<cbnum_B;blki++)
	{
        int colbnum=colblock_ptr_B[blki+1]-colblock_ptr_B[blki];
        SMatrix *subrowmatrixB_trans=(SMatrix *)malloc(colbnum*sizeof(SMatrix));
        int rowlength= blki==cbnum_B-1 ? colB-(cbnum_B-1)*BLOCK_SIZE : BLOCK_SIZE ;

     //   printf("rowlength=%d\n",rowlength);
        int start= blki*BLOCK_SIZE;
        int end = blki==cbnum_B-1 ?  colB : (blki+1)*BLOCK_SIZE ;
    
        for (int bi=0;bi<colbnum;bi++) 
        {
           subrowmatrixB_trans[bi].value=(MAT_VAL_TYPE*)malloc(( nnzb_B[colblock_ptr_B[blki]+bi+1] - nnzb_B[colblock_ptr_B[blki]+bi])*sizeof(MAT_VAL_TYPE));
           subrowmatrixB_trans[bi].columnindex=(int *)malloc(( nnzb_B[colblock_ptr_B[blki]+bi +1] - nnzb_B[colblock_ptr_B[blki]+bi])*sizeof(int));
          
           subrowmatrixB_trans[bi].rowpointer=(MAT_PTR_TYPE *)malloc((rowlength+1)*sizeof(MAT_PTR_TYPE));
           memset(subrowmatrixB_trans[bi].rowpointer,0,(rowlength+1)*sizeof(MAT_PTR_TYPE));
        }
        int *num=(int*)malloc((colbnum)*sizeof(int));
	    memset(num,0,(colbnum)*sizeof(int));

        for (int ri=0;ri<rowlength;ri++)
        {
            for (int j=cscColPtrB[start+ri];j<cscColPtrB[start+ri+1];j++)
            {
                int ki;
                for (int k=colblock_ptr_B[blki],ki=0;k<colblock_ptr_B[blki+1],ki<colbnum;k++,ki++)
                {
                    //int kcstart=blkcolIdx_B_trans[k]*BLOCK_SIZE;
                    int kcstart=blkrowIdx_B[k]*BLOCK_SIZE;
                    //int kcend= blkcolIdx_B_trans[k]== (cbnum_A-1) ?  colA: (blkcolIdx_B_trans[k]+1)*BLOCK_SIZE;
                    int kcend= blkrowIdx_B[k]== (cbnum_A-1) ?  colA: (blkrowIdx_B[k]+1)*BLOCK_SIZE;
                    if (cscRowIdxB[j]>=kcstart&&cscRowIdxB[j]<kcend)
                    {
                        num[ki]++;
                        subrowmatrixB_trans[ki].value[num[ki]-1]=cscValB[j];
                        //subrowmatrixB_trans[ki].columnindex[num[ki]-1]=cscRowIdxB[j]-blkcolIdx_B_trans[k]*BLOCK_SIZE;
                        subrowmatrixB_trans[ki].columnindex[num[ki]-1]=cscRowIdxB[j]-blkrowIdx_B[k]*BLOCK_SIZE;
                        break; 
                    }
                }
            }
            for (int bi=0;bi<colbnum;bi++){
                    subrowmatrixB_trans[bi].rowpointer[ri+1]=num[bi];
            }   
	    }
        //transpose submatrix

        SMatrix *subrowmatrixB=(SMatrix *)malloc(colbnum*sizeof(SMatrix));
        for (int bi=0;bi<colbnum;bi++) 
        {
            //int collength = blkcolIdx_B_trans[colblock_ptr_B[blki]+bi] == rbnum_B-1 ? rowB - (rbnum_B-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
            int collength = blkrowIdx_B[colblock_ptr_B[blki]+bi] == rbnum_B-1 ? rowB - (rbnum_B-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
             subrowmatrixB[bi].value=(MAT_VAL_TYPE*)malloc((nnzb_B[colblock_ptr_B[blki]+bi+1] - nnzb_B[colblock_ptr_B[blki]+bi])*sizeof(MAT_VAL_TYPE));
            subrowmatrixB[bi].columnindex=(int *)malloc(( nnzb_B[colblock_ptr_B[blki]+bi+1] - nnzb_B[colblock_ptr_B[blki]+bi])*sizeof(int));
          
            subrowmatrixB[bi].rowpointer=(MAT_PTR_TYPE *)malloc((collength+1)*sizeof(MAT_PTR_TYPE));
            memset(subrowmatrixB[bi].rowpointer,0,(collength+1)*sizeof(MAT_PTR_TYPE));
        }
        for (int bi =0;bi<colbnum; bi++)
        {
            //int collength = blkcolIdx_B_trans[colblock_ptr_B[blki]+bi] == rbnum_B-1 ? rowB - (rbnum_B-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
            int collength = blkrowIdx_B[colblock_ptr_B[blki]+bi] == rbnum_B-1 ? rowB - (rbnum_B-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
            matrix_transposition(rowlength,collength,nnzb_B[colblock_ptr_B[blki]+bi +1]-nnzb_B[colblock_ptr_B[blki]+bi],
                                subrowmatrixB_trans[bi].rowpointer,subrowmatrixB_trans[bi].columnindex,subrowmatrixB_trans[bi].value,
                                subrowmatrixB[bi].columnindex,subrowmatrixB[bi].rowpointer,subrowmatrixB[bi].value) ;
        }
      
    
        for(int bi=0;bi<colbnum;bi++)
        {
        
           //int collength = blkcolIdx_B_trans[colblock_ptr_B[blki]+bi] == rbnum_B-1 ? rowB - (rbnum_B-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
           int collength = blkrowIdx_B[colblock_ptr_B[blki]+bi] == rbnum_B-1 ? rowB - (rbnum_B-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
            //CSR val&col
/*
            for (int k=0;k<nnzb_B[colblock_ptr_B[blki] + bi + 1]-nnzb_B[colblock_ptr_B[blki] + bi];k++)
            {
                blkcsr_Val_B[nnzb_B[colblock_ptr_B[blki]+bi] +k]=subrowmatrixB[bi].value[k] ;
        //    Block_Val [blknnz[rowblock_ptr[blki] + bi] + k] = subrowmatrixA[bi].value[k] ;
                blkcsr_Col_B[nnzb_B[colblock_ptr_B[blki]+bi] +k]=subrowmatrixB[bi].columnindex[k];
            //    csrvid++;
 //mask[(colblock_ptr_B[blki] + bi) * BLOCK_SIZE + bri] |= (0x1 << (BLOCK_SIZE - colidx - 1));
            }
            //CSR ptr

            for (int jid=0;jid<collength;jid++)
            {
                blkcsr_Ptr_B[(colblock_ptr_B[blki] + bi) * BLOCK_SIZE + jid]=subrowmatrixB[bi].rowpointer[jid];
            //    csrpid++;
            }
*/
  for ( int bri = 0;bri <collength ;bri ++)
            {
                for (int k=subrowmatrixB[bi].rowpointer[bri] ;k < subrowmatrixB[bi].rowpointer[bri + 1];k++)
                {
                    int colidx = subrowmatrixB[bi].columnindex[k];
                    blkcsr_Val_B[nnzb_B[colblock_ptr_B[blki]+bi] +k]=subrowmatrixB[bi].value[k] ;


                    //blk_idx_B[nnzb_B[colblock_ptr_B[blki]+bi] +k] = (bri << 4) + colidx;

                    blkmaskB[(colblock_ptr_B[blki] + bi) * BLOCK_SIZE + bri] |= (0x1 << (BLOCK_SIZE - colidx - 1));
                //    res[count] = (str1[i] << 4) + str2[i];
                    blkcsr_Col_B[nnzb_B[colblock_ptr_B[blki]+bi ]+ k]=subrowmatrixB[bi].columnindex[k];

                }
                blkcsr_Ptr_B[(colblock_ptr_B[blki] + bi) * BLOCK_SIZE + bri]=subrowmatrixB[bi].rowpointer[bri];

            }

            for (int jid=collength;jid<BLOCK_SIZE;jid++)
            {
                blkcsr_Ptr_B[(colblock_ptr_B[blki] + bi) * BLOCK_SIZE + jid]=subrowmatrixB[bi].rowpointer[collength];
            //    csrpid++;
            }
        //    ptroffset_B += collength ;
        }
    
        for (int bi=0;bi<colbnum;bi++)
        {
            free(subrowmatrixB[bi].value);
            free(subrowmatrixB[bi].columnindex);
            free(subrowmatrixB[bi].rowpointer);
            free(subrowmatrixB_trans[bi].value);
            free(subrowmatrixB_trans[bi].columnindex);
            free(subrowmatrixB_trans[bi].rowpointer);
        }
        free(subrowmatrixB);
        free(subrowmatrixB_trans);
        free(num);
    }

/*printf("nnzA\n");
for (int i=0;i<rowblock_ptr_A[rbnum_A] + 1;i++)
{
    printf("%d    ",nnzb_A[i]);
}
printf("\n");
printf("\n");
printf("nnzB\n");

for (int i=0;i<colblock_ptr_B[cbnum_B] + 1;i++)
{
    printf("%d    ",nnzb_B[i]);
}
printf("\n");
printf("\n");
*/

/*    printf("aval\n");
    for (int i=0;i<nnzA;i++)
    {
        printf("%d     ",(int)blkcsr_Val_A[i]);
    }
    printf("\n");
    printf("\n");
*/
/*for ( int i = 0; i < nnzA; i++)
{
    printf("%d    ",matrixA.columnindex[i]);
}
        printf("\n");
        printf("\n");

        for ( int i = 0; i < rowA +1; i++)
{
    printf("%d    ",matrixA.rowpointer[i]);
}
        printf("\n");
        printf("\n");
*/


/*    printf("acol\n");
    for (int i=0;i<rowblock_ptr_A[rbnum_A];i++)
    {
        printf("The %d :  ",i);
        for (int j=nnzb_A[i];j<nnzb_A[i+1];j++)
        {
             printf("%d     ",(int)blkcsr_Col_A[j]);
        }
        printf("\n");
        printf("\n");
    }
    
    printf("\n");
    printf("aptr\n");
    printf("\n");
    for (int i=0;i<rowblock_ptr_A[rbnum_A];i++)
    {
         printf("The %d :  ",i);
        for (int j=BLOCK_SIZE * i;j<BLOCK_SIZE *(i+1);j++)
        {
             printf("%d     ",(int)blkcsr_Ptr_A[j]);
        }
        printf("\n");
        printf("\n");
    }


     printf("bcol\n");
    for (int i=0;i<colblock_ptr_B[cbnum_B];i++)
    {
        printf("The %d :  ",i);
        for (int j=nnzb_B[i];j<nnzb_B[i+1];j++)
        {
             printf("%d     ",(int)blkcsr_Col_B[j]);
        }
        printf("\n");
        printf("\n");
    }
    
    printf("\n");
    printf("bptr\n");
    printf("\n");
    for (int i=0;i<colblock_ptr_B[cbnum_B];i++)
    {
         printf("The %d :  ",i);
        for (int j=BLOCK_SIZE * i;j<BLOCK_SIZE *(i+1);j++)
        {
             printf("%d     ",(int)blkcsr_Ptr_B[j]);
        }
        printf("\n");
        printf("\n");
    }
*/

/*for (int i=0;i<rbnum_A +1;i++)
{
    printf("%d   ",rowblock_ptr_C[i]);
}
printf("\n") ;
for (int i=0;i<rowblock_ptr_C[rbnum_A];i++)
{
    printf("%d   ",blkcolIdx_C[i]);
}
printf("\n") ;
*/
/*for (int i=0;i<rowblock_ptr_A[rbnum_A]+ 1;i++)
{
    printf("%d    ",nnzb_A[i]);
}
printf("\n");
printf("\n");

for (int i=0;i<colblock_ptr_B[cbnum_B] + 1;i++)
{
    printf("%d    ",nnzb_B[i]);
}
printf("\n");
printf("\n");
*/


    int *nnzb_C = (int *)malloc((rowblock_ptr_C[rbnum_A] + 1) * sizeof (int)) ;
    memset(nnzb_C,0,(rowblock_ptr_C[rbnum_A] + 1) * sizeof (int)) ;

    int nnzC = 0;
    step3  (rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,nnzb_A,rowA,
            blkcsr_Val_A,blkcsr_Col_A,blkcsr_Ptr_A,
            colblock_ptr_B,blkrowIdx_B,rbnum_B,cbnum_B,nnzb_B,
            blkcsr_Val_B,blkcsr_Col_B,blkcsr_Ptr_B,
            rowblock_ptr_C,blkcolIdx_C,nnzb_C, &nnzC);

    unsigned char *d_blkcsr_Ptr_C;
    cudaMalloc((void **)&d_blkcsr_Ptr_C, numblkC * BLOCK_SIZE * sizeof(unsigned char));
    //cudaMemset(d_blkcsr_Ptr_C, 0, numblkC * BLOCK_SIZE * sizeof(unsigned char));

    unsigned short *d_blkmaskC;
    cudaMalloc((void **)&d_blkmaskC, numblkC *BLOCK_SIZE * sizeof(unsigned short));
    //cudaMemset(d_blkmaskC, 0, numblkC *BLOCK_SIZE * sizeof(unsigned short));

    int *d_blkid_smem;
    int *d_blkid_dns;
    cudaMalloc((void **)&d_blkid_smem, numblkC  * sizeof(int));
    cudaMalloc((void **)&d_blkid_dns, numblkC * sizeof(int));

    int *d_blksmem_cnt;
    int *d_blkdns_cnt;

    cudaMalloc((void **)&d_blksmem_cnt, 1 * sizeof(int));
    cudaMalloc((void **)&d_blkdns_cnt, 1 * sizeof(int));

    cudaMemset(d_blksmem_cnt, 0, 1 * sizeof(int));
    cudaMemset(d_blkdns_cnt, 0, 1 * sizeof(int));

    step3_cuda  (rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,numblkA, nnzA, nnzb_A,rowA,
            blkcsr_Val_A,blk_idx_A,blkcsr_Ptr_A,blkmaskA,
            colblock_ptr_B,blkrowIdx_B,rbnum_B,cbnum_B,numblkB, nnzB, nnzb_B,
            blkcsr_Val_B,blkcsr_Col_B,blkcsr_Ptr_B,blkmaskB,
            rowblock_ptr_C,blkcolIdx_C, d_blkrowIdx_C, d_blkcolIdx_C, d_blkcsr_Ptr_C, d_blkmaskC, 
            nnzb_C, numblkC, &nnzC, d_blksmem_cnt, d_blkdns_cnt, d_blkid_smem, d_blkid_dns);



    
    printf("nnzC = %i\n",nnzC) ;

    printf("step 3 success!\n");


    MAT_VAL_TYPE *blkcsr_Val_C=(MAT_VAL_TYPE*)malloc((nnzC)*sizeof(MAT_VAL_TYPE));
    unsigned char  *blkcsr_Col_C=(unsigned char*)malloc((nnzC)*sizeof(unsigned char));
    unsigned char *blkcsr_Ptr_C=(unsigned char*)malloc((BLOCK_SIZE * numblkC)*sizeof(unsigned char));
    memset(blkcsr_Ptr_C,0,(BLOCK_SIZE * numblkC)*sizeof(char)) ;

     step4  (rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,nnzb_A,rowA,
            blkcsr_Val_A,blkcsr_Col_A,blkcsr_Ptr_A,
            colblock_ptr_B,blkrowIdx_B,rbnum_B,cbnum_B,nnzb_B,
            blkcsr_Val_B,blkcsr_Col_B,blkcsr_Ptr_B,
            rowblock_ptr_C,blkcolIdx_C,nnzb_C,
            blkcsr_Val_C,blkcsr_Col_C,blkcsr_Ptr_C);

unsigned char *d_blkcsr_Col_C;
    cudaMalloc((void **)&d_blkcsr_Col_C, nnzC * sizeof(unsigned char));
MAT_VAL_TYPE *d_blkcsr_Val_C;
    cudaMalloc((void **)&d_blkcsr_Val_C, nnzC * sizeof(MAT_VAL_TYPE));

     step4_cuda  (rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,numblkA, nnzA, nnzb_A,rowA,
            blkcsr_Val_A,blk_idx_A,blkcsr_Ptr_A,
            colblock_ptr_B,blkrowIdx_B,rbnum_B,cbnum_B,numblkB, nnzB, nnzb_B,
            blkcsr_Val_B,blkcsr_Col_B,blkcsr_Ptr_B,
            rowblock_ptr_C,blkcolIdx_C,
d_blkrowIdx_C, d_blkcolIdx_C,
d_blkcsr_Ptr_C, d_blkcsr_Col_C, d_blkcsr_Val_C, d_blkmaskC,
nnzb_C,numblkC,
            blkcsr_Val_C,blkcsr_Col_C,blkcsr_Ptr_C, nnzC, 
d_blksmem_cnt, d_blkdns_cnt, d_blkid_smem, d_blkid_dns);

    printf("step 4 success!\n");
//return 0;

    printf("\n\nstepall_cuda_new started!\n");
unsigned long long int nnzC_computed;
    double compression_rate = 0;
double time_stir = 0;
    double gflops_stir = 0;
 stepall_cuda_new (rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,numblkA, nnzA, nnzb_A,rowA,
            blkcsr_Val_A,blk_idx_A,blkcsr_Ptr_A,blkmaskA,
            colblock_ptr_B,blkrowIdx_B,rowblock_ptr_B,blkcolIdx_B,rbnum_B,cbnum_B,numblkB, nnzB, nnzb_B,
            blkcsr_Val_B,blkcsr_Col_B,blkcsr_Ptr_B,blkmaskB,
rowblock_ptr_C,blkcolIdx_C,
nnzb_C,numblkC,
            blkcsr_Val_C,blkcsr_Col_C,blkcsr_Ptr_C, nnzC, nnzCub, 
&nnzC_computed, &compression_rate, &time_stir, &gflops_stir);
    printf("stepall_cuda_new success!\n");

    // write results to text (scv) file
    FILE *fout = fopen("results.csv", "a");
    if (fout == NULL)
        printf("Writing results fails.\n");
    fprintf(fout, "%s,%i,%i,%i,%lld,%lld,%f,%f,%f\n",
            filename, rowA, colA, nnzA, nnzCub, nnzC_computed, compression_rate, time_stir, gflops_stir);
    fclose(fout);


/*    for(int i=0;i<nnzC ;i++)
    {
        printf("%d    ",(int)blkcsr_Val_C[i]);
    }
    printf("\n");
     printf("\n");

    for(int i=0;i<nnzC ;i++)
    {
        printf("%d    ",blkcsr_Col_C[i]);
    }
    printf("\n");
     printf("\n");

    for (int i=0;i<BLOCK_SIZE * rbnum_A * cbnum_B ; i++)
    {
        printf("%d    ",blkcsr_Ptr_C[i]);
    }
     printf("\n");
     printf("\n");
*/

cudaFree(d_blkrowIdx_C);
cudaFree(d_blkcolIdx_C);



}

    
