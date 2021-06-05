#include"common.h"
#include"mmio_highlevel.h"
//#include"mmio.h"
#include"utils.h"
//#include "encode.h"
#include"utils_cuda_scan.h"

#include "spgemm_nsparse_kernel.h"

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "cuda.h"
//#include <cuda_runtime_api.h>
//#include "cuda_runtime.h"

# define INDEX_DATA_TYPE unsigned char
//# define VAL_DATA_TYPE double

//#define WARMUP_NUM 3
#define REPEAT_NUM 10
#define result_check 1
#define DEBUG 1

#define AAT 0

#define SMEM_TNY_TH 32
#define SMEM_SML_TH 48
#define SMEM_LRG_TH 192
#define SMEM_DNS_TH 256

#define TILE_PER_WARP 16 // should not be larger than WARPSIZE


typedef struct 
{
	MAT_VAL_TYPE *value;
	int *columnindex;
	MAT_PTR_TYPE *rowpointer;
}SMatrix;

int carveout = cudaSharedmemCarveoutMaxShared; // (100)

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
int binary_search_exact_kernel(const int * __restrict__ d_array, int l, int r, int key) 
{ 
    while (l <= r) { 
        int m = l + (r - l) / 2; 
        int elem = ld_gbl_int32(d_array+m);
        // Check if x is present at mid 
        if (elem == key) 
            return m; 
  
        // If x greater, ignore left half 
        if (elem < key) 
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
int binary_search_exact_uchar_kernel(const unsigned char *d_array, int l, int r, unsigned char key) 
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
int binary_search_right_boundary_kernel(const int * __restrict__ d_row_pointer,
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
        key_median = ld_gbl_int32(d_row_pointer+median);
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
int *cuda_memory_use; 
double *time_node;
int *index;

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
            int *d_blkrowptrC, int *blkrowptrC_golden, int *numblkC) 
{
//for (int i = 0; i < blkmA; i++)
//    printf("[%i],%i\n", i, blkrowptrA[i+1] - blkrowptrA[i]);

// struct timeval tv;

int *d_blkrowptrA;
 int *d_blkcolidxA; 
            int *d_blkrowptrB; int *d_blkcolidxB;
         //   int *d_blkrowptrC;

    cudaMalloc((void **)&d_blkrowptrA, (blkmA+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // // printf("time = %d\n", time_node[*index]);
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (blkmA+1) * sizeof(int);
    // printf("\n\n\nindex = %d    mem = %d    time = %d\n\n\n\n", *index, cuda_memory_use[*index], time_node[*index]);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcolidxA, numblkA * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkA * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkrowptrB, (blkmB+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (blkmB+1) * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcolidxB, numblkB * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkB * sizeof(int);
    // (*index) += 1;

    // cudaMalloc((void **)&d_blkrowptrC, (blkmA+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (blkmA+1) * sizeof(int);
    // (*index) += 1;

    //cudaMemset(d_blkrowptrC, 0, (blkmA+1) * sizeof(int));

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
                numblkC
                );

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
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (blkmA+1) * sizeof(int);
    // (*index) += 1;

    cudaFree(d_blkcolidxA);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkA * sizeof(int);
    // (*index) += 1;

    cudaFree(d_blkrowptrB);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (blkmB+1) * sizeof(int);
    // (*index) += 1;

    cudaFree(d_blkcolidxB);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkB * sizeof(int);
    // (*index) += 1;

   // cudaFree(d_blkrowptrC);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (blkmA+1) * sizeof(int);
    // (*index) += 1;    

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
        //int blknnz_C =0;
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
    //__shared__ int s_numblkC[1];

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
// struct timeval tv;
int *d_blkrowptrA;
 int *d_blkcolidxA; 
            int *d_blkrowptrB; int *d_blkcolidxB;
            int *d_blkrowptrC;
            int *d_blkrowptrC_offset;

    cudaMalloc((void **)&d_blkrowptrA, (blkmA+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (blkmA+1) * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcolidxA, numblkA * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkA * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkrowptrB, (blkmB+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (blkmB+1) * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcolidxB, numblkB * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkB * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkrowptrC, (blkmA+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (blkmA+1) * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkrowptrC_offset, (blkmA+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (blkmA+1) * sizeof(int);
    // (*index) += 1;


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
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (blkmA+1) * sizeof(int);
    // (*index) += 1;

    cudaFree(d_blkcolidxA);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkA * sizeof(int);
    // (*index) += 1;

    cudaFree(d_blkrowptrB);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (blkmB+1) * sizeof(int);
    // (*index) += 1;

    cudaFree(d_blkcolidxB);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkB * sizeof(int);
    // (*index) += 1;

    cudaFree(d_blkrowptrC);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (blkmA+1) * sizeof(int);
    // (*index) += 1;        

    cudaFree(d_blkrowptrC_offset);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (blkmA+1) * sizeof(int);
    // (*index) += 1; 

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

template <int SMEM_SIZE, int THREADS_PER_ROW>
__global__
void stir_spgemm_step3_cuda_kernel
                             (const int* __restrict__ d_blkrowptrA,
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
                            int *d_blksmem_tny_cnt, 
                            int *d_blksmem_sml_cnt, 
                            int *d_blksmem_lrg_cnt, 
                            int *d_blksmem_dns_cnt, 
                            int *d_blkid_smem_tny,
                            int *d_blkid_smem_sml,
                            int *d_blkid_smem_lrg,
                            int *d_blkid_smem_dns,
                            int numblkC)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp_id = global_id >> 5; //global_id / WARP_SIZE;

    if (global_warp_id >= numblkC) return;

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
    const int local_warp_id = threadIdx.x >> 5; //threadIdx.x / WARP_SIZE;
    
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
    int lena = d_blkrowptrA[blki+1] - abase;

    const int bbase = d_blkcolptrB[blkj];
    const int bstop = d_blkcolptrB[blkj+1];
    int lenb = d_blkcolptrB[blkj+1] - bbase;


    if (lena < lenb)
    {
        for (int i = lane_id; i < lena; i+= WARP_SIZE)
        {
            int idxa = ld_gbl_int32(d_blkcolidxA+abase+i);
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
                
/*
            // sub-warp method (process row/num_subwarp one time, each row uses THREADS_PER_ROW threads)
            //const int THREADS_PER_ROW = 4;
            const int num_subwarp = WARP_SIZE / THREADS_PER_ROW;
            const int sublane_id = lane_id % THREADS_PER_ROW;
            const int subwarp_id = lane_id / THREADS_PER_ROW;

            const int nnzastart = d_nnzb_A[(abase+posa)];
            int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;

            for (int rowidxa = subwarp_id; rowidxa < BLOCK_SIZE; rowidxa += num_subwarp)
            {
                int blkstarta = d_blkcsr_Ptr_A[(abase+posa)*BLOCK_SIZE+rowidxa];
                int blkstopa = rowidxa == BLOCK_SIZE - 1 ? nnztotala : d_blkcsr_Ptr_A[(abase+posa)*BLOCK_SIZE+rowidxa+1];

                for (int j = blkstarta; j < blkstopa; j++)
                {
                    unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+j];
                   unsigned int maskb = s_blkmaskB_local[rowcolidx & 0xf];
                   s_maskc_local[rowcolidx >> 4] |= maskb;
                }
            }
            */
        }
    }
    else 
    {
        const int astart = d_blkcolidxA[abase];
        const int aend = d_blkcolidxA[astop-1];
        const int bstart = d_blkrowidxB[bbase];
        const int bend = d_blkrowidxB[bstop-1];

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
        if (nnzcnt <= SMEM_DNS_TH)// && nnzcnt > 0)
        {
            int pos = atomicAdd(d_blksmem_cnt, 1);
            d_blkid_smem[pos] = global_warp_id;
        }
        else if (nnzcnt > SMEM_DNS_TH)
        {
            int pos = atomicAdd(d_blkdns_cnt, 1);
            d_blkid_dns[pos] = global_warp_id;
        }
*/
        if (nnzcnt <= SMEM_TNY_TH && nnzcnt > 0)
        {
            int pos = atomicAdd(d_blksmem_tny_cnt, 1);
            d_blkid_smem_tny[pos] = global_warp_id;
        }
        else if (SMEM_TNY_TH < nnzcnt && nnzcnt <= SMEM_SML_TH)
        {
            int pos = atomicAdd(d_blksmem_sml_cnt, 1);
            d_blkid_smem_sml[pos] = global_warp_id;
        }
        else if (SMEM_SML_TH < nnzcnt && nnzcnt <= SMEM_LRG_TH)
        {
            int pos = atomicAdd(d_blksmem_lrg_cnt, 1);
            d_blkid_smem_lrg[pos] = global_warp_id;
        }
        else if (SMEM_LRG_TH < nnzcnt && nnzcnt <= SMEM_DNS_TH)
        {
            int pos = atomicAdd(d_blksmem_dns_cnt, 1);
            d_blkid_smem_dns[pos] = global_warp_id;
        }
    }
}
}


template <int SMEM_SIZE, int TPW>
__global__
void stir_spgemm_step3_cuda_kernel_2level
                           (const int* d_blkrowptrA,
                            const int* __restrict__ d_blkcolidxA,
                            const int *d_nnzb_A,
                            MAT_VAL_TYPE *d_blkcsr_Val_A,
                            unsigned char *d_blkcsr_Col_A,
                            unsigned char *d_blkcsr_Ptr_A,
                            unsigned short *d_blkmaskA,
                            int blkmA, int blknA, int numblkA, int nnzA, 
                            const int* __restrict__ d_blkcolptrB,
                            const int* __restrict__ d_blkrowidxB, 
                            const int* __restrict__ d_nnzb_B,
                            const MAT_VAL_TYPE* __restrict__ d_blkcsr_Val_B,
                            const unsigned char* __restrict__ d_blkcsr_Col_B,
                            const unsigned char* __restrict__ d_blkcsr_Ptr_B,
                            const unsigned short* __restrict__ d_blkmaskB,
                            int blkmB, int blknB , int numblkB, int nnzB, 
                            int *d_blkrowidxC,
                            int *d_blkcolidxC,
                            unsigned char *d_blkcsr_Ptr_C,
                            int *d_nnzb_C,
                            unsigned short *d_blkmaskC,
                            int *d_blksmem_tny_cnt, 
                            int *d_blksmem_sml_cnt, 
                            int *d_blksmem_lrg_cnt, 
                            int *d_blksmem_dns_cnt, 
                            int *d_blkid_smem_tny,
                            int *d_blkid_smem_sml,
                            int *d_blkid_smem_lrg,
                            int *d_blkid_smem_dns,
                            int numblkC)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp_id = global_id >> 5; //global_id / WARP_SIZE;

    __shared__ unsigned int s_maskc[WARP_PER_BLOCK * BLOCK_SIZE];
    //__shared__ unsigned short s_blkmaskA[WARP_PER_BLOCK * BLOCK_SIZE];
    __shared__ unsigned short s_blkmaskB[WARP_PER_BLOCK * BLOCK_SIZE];

    __shared__ int s_matched_posa[WARP_PER_BLOCK * SMEM_SIZE];
    __shared__ int s_matched_posb[WARP_PER_BLOCK * SMEM_SIZE];
    __shared__ int s_matchedcnt[WARP_PER_BLOCK];

    __shared__ int s_blksmem_tny_cnt[WARP_PER_BLOCK];
    __shared__ int s_blksmem_sml_cnt[WARP_PER_BLOCK];
    __shared__ int s_blksmem_lrg_cnt[WARP_PER_BLOCK];
    __shared__ int s_blksmem_dns_cnt[WARP_PER_BLOCK];

    __shared__ int s_blkid_smem_tny[WARP_PER_BLOCK * TPW];
    __shared__ int s_blkid_smem_sml[WARP_PER_BLOCK * TPW];
    __shared__ int s_blkid_smem_lrg[WARP_PER_BLOCK * TPW];
    __shared__ int s_blkid_smem_dns[WARP_PER_BLOCK * TPW];

    //const int local_warp_id = threadIdx.x / WARP_SIZE;
    //__shared__ char s_dnsidx[WARP_PER_BLOCK * BLOCK_SIZE * BLOCK_SIZE];
    //char *s_dnsidx_local = &s_dnsidx[local_warp_id * BLOCK_SIZE * BLOCK_SIZE];

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    //const int sublane_id = lane_id % (WARP_SIZE / BLOCK_SIZE);
    //const int subwarp_id = lane_id / (WARP_SIZE / BLOCK_SIZE);
    
    const int local_warp_id = threadIdx.x >> 5; //threadIdx.x / WARP_SIZE;
    
    unsigned int *s_maskc_local = &s_maskc[local_warp_id * BLOCK_SIZE];
    //unsigned short *s_blkmaskA_local = &s_blkmaskA[local_warp_id * BLOCK_SIZE];
    unsigned short *s_blkmaskB_local = &s_blkmaskB[local_warp_id * BLOCK_SIZE];

    int *s_matched_posa_local = &s_matched_posa[local_warp_id * SMEM_SIZE];
    int *s_matched_posb_local = &s_matched_posb[local_warp_id * SMEM_SIZE];
    int *s_matchedcnt_local = &s_matchedcnt[local_warp_id];

    int *s_blksmem_tny_cnt_local = &s_blksmem_tny_cnt[local_warp_id];
    int *s_blksmem_sml_cnt_local = &s_blksmem_sml_cnt[local_warp_id];
    int *s_blksmem_lrg_cnt_local = &s_blksmem_lrg_cnt[local_warp_id];
    int *s_blksmem_dns_cnt_local = &s_blksmem_dns_cnt[local_warp_id];

    int *s_blkid_smem_tny_local = &s_blkid_smem_tny[local_warp_id * TPW];
    int *s_blkid_smem_sml_local = &s_blkid_smem_sml[local_warp_id * TPW];
    int *s_blkid_smem_lrg_local = &s_blkid_smem_lrg[local_warp_id * TPW];
    int *s_blkid_smem_dns_local = &s_blkid_smem_dns[local_warp_id * TPW];

    //if (global_warp_id >= numblkC) return;
    //if (global_warp_id < numblkC)
    int tile_start = global_warp_id * TPW;
    if (tile_start >= numblkC) return;

    int tile_end = tile_start + TPW; //(global_warp_id + 1) * TPW;
    tile_end = tile_end >= numblkC ? numblkC : tile_end;

    if (!lane_id) 
    {
        s_blksmem_tny_cnt_local[0] = 0;
        s_blksmem_sml_cnt_local[0] = 0;
        s_blksmem_lrg_cnt_local[0] = 0;
        s_blksmem_dns_cnt_local[0] = 0;
    }

    for (int tilei = tile_start; tilei < tile_end; tilei++)
    {
        //unsigned int maskc = 0;

           //for (int i = 0; i < BLOCK_SIZE; i++)
        //    if (lane_id < BLOCK_SIZE)
        //        s_dnsidx_local[i * BLOCK_SIZE + lane_id] = 0;
        //__syncthreads();
        if (lane_id < BLOCK_SIZE)
            s_maskc_local[lane_id] = 0;
        if (!lane_id) 
            s_matchedcnt_local[0] = 0;
        __syncthreads();

        const int blki = d_blkrowidxC[tilei];
        const int blkj = d_blkcolidxC[tilei];

        const int abase = d_blkrowptrA[blki];
        const int astop = d_blkrowptrA[blki+1];
        int lena = astop - abase;

        const int bbase = ld_gbl_int32(d_blkcolptrB+blkj);
        const int bstop = ld_gbl_int32(d_blkcolptrB+blkj+1);
        int lenb = bstop - bbase;

        if (lena < lenb)
        {
            for (int i = lane_id; i < lena; i+= WARP_SIZE)
            {
                int idxa = d_blkcolidxA[abase+i];
                int res = binary_search_exact_kernel(d_blkrowidxB+bbase, 0, lenb-1, idxa);
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
                int idxb = ld_gbl_int32(d_blkrowidxB+bbase+i);
                int res = binary_search_exact_kernel(d_blkcolidxA+abase, 0, lena-1, idxb);
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
                    s_blkmaskB_local[lane_id] = ld_gbl_ushort(&d_blkmaskB[(bbase+posb) * BLOCK_SIZE + lane_id]);
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
            const int astart = d_blkcolidxA[abase];
            const int aend = d_blkcolidxA[astop-1];
            const int bstart = ld_gbl_int32(d_blkrowidxB+bbase);
            const int bend = ld_gbl_int32(d_blkrowidxB+bstop-1);

            int posa_real = 0;
            int posb_real = 0;

            if (bstart > astart)
            {
                //posa_real = binary_search_right_boundary_kernel((lena <= 128 ? s_blkcolidxA : &d_blkcolidxA[abase]), bstart, lena);
                //posa_real = binary_search_right_boundary_kernel(&d_blkcolidxA[abase], bstart, lena);
                int posa_real_new = binary_search_right_boundary_kernel(d_blkcolidxA+abase, bstart, lena);
                posa_real = posa_real_new < 0 ? 0 : posa_real_new;
            }
            else if (bstart < astart)
            {
                //posb_real = binary_search_right_boundary_kernel(&d_blkrowidxB[bbase], astart, lenb);
                int posb_real_new = binary_search_right_boundary_kernel(d_blkrowidxB+bbase, astart, lenb);
                posb_real = posb_real_new < 0 ? 0 : posb_real_new;
            }

            if (bstop < astop)
            {
                int lena_new = binary_search_right_boundary_kernel(d_blkcolidxA+abase, bend, lena) + 1;
                lena = lena_new > lena ? lena : lena_new;
            }
            else if (bstop > astop)
            {
                int lenb_new = binary_search_right_boundary_kernel(d_blkrowidxB+bbase, aend, lenb) + 1;
                lenb = lenb_new > lenb ? lenb : lenb_new;
            }
            for (int posa = 0; posa < lena; posa++)
            {
                int idxa = d_blkcolidxA[abase+posa]; 
                int posb = binary_search_right_boundary_kernel(d_blkrowidxB+bbase+posb_real, idxa, lenb-posb_real);
                if (posb < 0) continue;
                if (posb > lenb-posb_real) break;
                int idxb = ld_gbl_int32(d_blkrowidxB+bbase+posb_real+posb);

                if (idxa == idxb)
                {
                    posb_real = posb_real + posb;
                    if (lane_id < BLOCK_SIZE)
                    {
                        s_blkmaskB_local[lane_id] = ld_gbl_ushort(d_blkmaskB+(bbase+posb_real) * BLOCK_SIZE + lane_id);
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

        unsigned int maskc = lane_id < BLOCK_SIZE ? s_maskc_local[lane_id] : 0;
        int nnzcnt = __popc(maskc); //lane_id < BLOCK_SIZE ? __popc(maskc) : 0;

        int nnzcnt_sum = sum_32_shfl(nnzcnt);

        if (nnzcnt_sum == 0)
        {
            if (lane_id < BLOCK_SIZE) 
            {
                d_blkcsr_Ptr_C[tilei * BLOCK_SIZE + lane_id] = 0;
                d_blkmaskC[tilei * BLOCK_SIZE + lane_id] = 0;
            }
            if (!lane_id)
                d_nnzb_C[tilei] = 0;
        }
        else
        {
            int nnzcnt_scan = scan_32_shfl(nnzcnt, lane_id);
            if (lane_id < BLOCK_SIZE) 
                d_blkcsr_Ptr_C[tilei * BLOCK_SIZE + lane_id] = nnzcnt_scan - nnzcnt;

            nnzcnt = __shfl_sync(0xffffffff, nnzcnt_scan, BLOCK_SIZE);

            if (lane_id < BLOCK_SIZE && nnzcnt)
                d_blkmaskC[tilei * BLOCK_SIZE + lane_id] = s_maskc_local[lane_id];

            if (!lane_id)
            {
                d_nnzb_C[tilei] = nnzcnt;

                //printf("%i,%i\n", tilei, nnzcnt);
                if (nnzcnt <= SMEM_TNY_TH && nnzcnt > 0)
                {
                    //int pos = atomicAdd(d_blksmem_tny_cnt, 1);
                    //d_blkid_smem_tny[pos] = tilei;
                    int pos = atomicAdd(s_blksmem_tny_cnt_local, 1);
                    s_blkid_smem_tny_local[pos] = tilei;
                }
                else if (SMEM_TNY_TH < nnzcnt && nnzcnt <= SMEM_SML_TH)
                {
                    //int pos = atomicAdd(d_blksmem_sml_cnt, 1);
                    //d_blkid_smem_sml[pos] = tilei;
                    int pos = atomicAdd(s_blksmem_sml_cnt_local, 1);
                    s_blkid_smem_sml_local[pos] = tilei;
                }
                else if (SMEM_SML_TH < nnzcnt && nnzcnt <= SMEM_LRG_TH)
                {
                    //int pos = atomicAdd(d_blksmem_lrg_cnt, 1);
                    //d_blkid_smem_lrg[pos] = tilei;
                    int pos = atomicAdd(s_blksmem_lrg_cnt_local, 1);
                    s_blkid_smem_lrg_local[pos] = tilei;
                }
                else if (SMEM_LRG_TH < nnzcnt && nnzcnt <= SMEM_DNS_TH)
                {
                    //int pos = atomicAdd(d_blksmem_dns_cnt, 1);
                    //d_blkid_smem_dns[pos] = tilei;
                    int pos = atomicAdd(s_blksmem_dns_cnt_local, 1);
                    s_blkid_smem_dns_local[pos] = tilei;
                }
            }
        }
    }

    int len = s_blksmem_tny_cnt_local[0];
    int pos = lane_id == 0 ? atomicAdd(d_blksmem_tny_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, 0);
    if (lane_id < len)
        d_blkid_smem_tny[pos + lane_id] = s_blkid_smem_tny_local[lane_id];

    len = s_blksmem_sml_cnt_local[0];
    pos = lane_id == 0 ? atomicAdd(d_blksmem_sml_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, 0);
    if (lane_id < len)
        d_blkid_smem_sml[pos + lane_id] = s_blkid_smem_sml_local[lane_id];

    len = s_blksmem_lrg_cnt_local[0];
    pos = lane_id == 0 ? atomicAdd(d_blksmem_lrg_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, 0);
    if (lane_id < len)
        d_blkid_smem_lrg[pos + lane_id] = s_blkid_smem_lrg_local[lane_id];

    len = s_blksmem_dns_cnt_local[0];
    pos = lane_id == 0 ? atomicAdd(d_blksmem_dns_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, 0);
    if (lane_id < len)
        d_blkid_smem_dns[pos + lane_id] = s_blkid_smem_dns_local[lane_id];
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
    const int global_warp_id = global_id >> 5; //global_id / WARP_SIZE;

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
    const int local_warp_id = threadIdx.x >> 5; //threadIdx.x / WARP_SIZE;
    
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
    const int global_warp_id = global_id >> 5; //global_id / WARP_SIZE;

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
    const int local_warp_id = threadIdx.x >> 5; //threadIdx.x / WARP_SIZE;
    
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
            int *d_nnzb_C, int *nnzb_C_golden,  int numblkC, int *nnzC, int *d_blksmem_tny_cnt, int *d_blksmem_sml_cnt, int *d_blksmem_lrg_cnt, int *d_blksmem_dns_cnt, 
            int *d_blkid_smem_tny, int *d_blkid_smem_sml, int *d_blkid_smem_lrg, int *d_blkid_smem_dns)
{
    // struct timeval tv;

    int *d_blkrowptrA;
    int *d_blkcolidxA; 
    int *d_nnzb_A; 
    MAT_VAL_TYPE *d_blkcsr_Val_A;
    unsigned char *d_blkcsr_Col_A;
    unsigned char *d_blkcsr_Ptr_A;
    unsigned short *d_blkmaskA;

    cudaMalloc((void **)&d_blkrowptrA, (blkmA+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (blkmA+1) * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcolidxA, numblkA * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkA * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_nnzb_A, (numblkA+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (numblkA+1) * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcsr_Val_A, nnzA * sizeof(MAT_VAL_TYPE));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + nnzA * sizeof(MAT_VAL_TYPE);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcsr_Col_A, nnzA * sizeof(unsigned char));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + nnzA * sizeof(unsigned char);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcsr_Ptr_A, numblkA * BLOCK_SIZE  * sizeof(unsigned char));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkA * BLOCK_SIZE  * sizeof(unsigned char);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkmaskA, numblkA * BLOCK_SIZE  * sizeof(unsigned short));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkA * BLOCK_SIZE  * sizeof(unsigned short);
    // (*index) += 1;


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
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (blknB+1) * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkrowidxB, numblkB * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkB * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_nnzb_B, (numblkB+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (numblkB+1) * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcsr_Val_B, nnzB * sizeof(MAT_VAL_TYPE));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + nnzB * sizeof(MAT_VAL_TYPE);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcsr_Col_B, nnzB * sizeof(unsigned char));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + nnzB * sizeof(unsigned char);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcsr_Ptr_B, numblkB * BLOCK_SIZE  * sizeof(unsigned char));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkB * BLOCK_SIZE  * sizeof(unsigned char);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkmaskB, numblkB * BLOCK_SIZE  * sizeof(unsigned short));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkB * BLOCK_SIZE  * sizeof(unsigned short);
    // (*index) += 1;

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
  //  int *d_nnzb_C;

    //cudaMalloc((void **)&d_blkrowptrC, (blkmA+1) * sizeof(int));
    //cudaMalloc((void **)&d_blkrowidxC, numblkC * sizeof(int));
    //cudaMalloc((void **)&d_blkcolidxC, numblkC * sizeof(int));
  //  cudaMalloc((void **)&d_nnzb_C, (numblkC+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (numblkC+1) * sizeof(int);
    // (*index) += 1;



    //cudaMemcpy(d_blkrowptrC,     blkrowptrC,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemcpy(d_blkrowidxC,     blkrowidxC,     numblkC * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemcpy(d_blkcolidxC,     blkcolidxC,     numblkC * sizeof(int),              cudaMemcpyHostToDevice);
  //  cudaMemset(d_nnzb_C, 0, (numblkC+1) * sizeof(int));


    struct timeval t1, t2;

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

    int num_threads = WARP_SIZE * WARP_PER_BLOCK;
/*
    int num_blocks = ceil((double)numblkC/(double)WARP_PER_BLOCK);
    stir_spgemm_step3_cuda_kernel<128,4><<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A, d_blkmaskA,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B, d_blkmaskB,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, d_nnzb_C, d_blkmaskC, 
                              d_blksmem_tny_cnt, d_blksmem_sml_cnt, d_blksmem_lrg_cnt, d_blksmem_dns_cnt, 
                              d_blkid_smem_tny, d_blkid_smem_sml, d_blkid_smem_lrg, d_blkid_smem_dns, numblkC);
*/
    int num_blocks = ceil((double)numblkC/(double)(WARP_PER_BLOCK * TILE_PER_WARP));
    stir_spgemm_step3_cuda_kernel_2level<128, TILE_PER_WARP><<< num_blocks, num_threads >>>
                    (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A, d_blkmaskA,
                    blkmA, blknA, numblkA, nnzA, 
                    d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B, d_blkmaskB,
                    blkmB, blknB, numblkB, nnzB, 
                    d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, d_nnzb_C, d_blkmaskC, 
                    d_blksmem_tny_cnt, d_blksmem_sml_cnt, d_blksmem_lrg_cnt, d_blksmem_dns_cnt, 
                    d_blkid_smem_tny, d_blkid_smem_sml, d_blkid_smem_lrg, d_blkid_smem_dns, numblkC);

exclusive_scan_device_cuda(d_nnzb_C, numblkC+1);
int nbc = 0;
cudaMemcpy(&nbc,     &d_nnzb_C[numblkC],     sizeof(int), cudaMemcpyDeviceToHost);
//*numblkC = nbc;
*nnzC = nbc;

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

/*
int blksmem_cnt = 0;
int blkdns_cnt = 0;
cudaMemcpy(&blksmem_cnt,     d_blksmem_cnt,     sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(&blkdns_cnt,     d_blkdns_cnt,     sizeof(int), cudaMemcpyDeviceToHost);

printf("blksmem_cnt = %i, blkdns_cnt = %i\n", blksmem_cnt, blkdns_cnt);
*/
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

    cudaFree(d_blkmaskA);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkA * BLOCK_SIZE  * sizeof(unsigned short);
    // (*index) += 1;                              

    cudaFree(d_blkrowptrA);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (blkmA+1) * sizeof(int);
    // (*index) += 1;                        

    cudaFree(d_blkcolidxA); 
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkA * sizeof(int);
    // (*index) += 1; 

    cudaFree(d_nnzb_A); 
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (numblkA+1) * sizeof(int);
    // (*index) += 1; 

    cudaFree(d_blkcsr_Val_A);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - nnzA * sizeof(MAT_VAL_TYPE);
    // (*index) += 1;                 

    cudaFree(d_blkcsr_Col_A);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - nnzA * sizeof(unsigned char);
    // (*index) += 1;                              

    cudaFree(d_blkcsr_Ptr_A);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkA * BLOCK_SIZE  * sizeof(unsigned char);
    // (*index) += 1;

    cudaFree(d_blkcolptrB);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (blknB+1) * sizeof(int);
    // (*index) += 1;                      

    cudaFree(d_blkrowidxB); 
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkB * sizeof(int);
    // (*index) += 1;                            

    cudaFree(d_nnzb_B); 
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (numblkB+1) * sizeof(int);
    // (*index) += 1;                           

    cudaFree(d_blkcsr_Val_B);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - nnzB * sizeof(MAT_VAL_TYPE);
    // (*index) += 1;                             

    cudaFree(d_blkcsr_Col_B);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - nnzB * sizeof(unsigned char);
    // (*index) += 1;                            

    cudaFree(d_blkcsr_Ptr_B);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkB * BLOCK_SIZE  * sizeof(unsigned char);
    // (*index) += 1;                            

    cudaFree(d_blkmaskB);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkB * BLOCK_SIZE  * sizeof(unsigned short);
    // (*index) += 1;

    //cudaFree(d_blkrowptrC);
    //cudaFree(d_blkrowidxC);
    //cudaFree(d_blkcolidxC);
   // cudaFree(d_nnzb_C);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (numblkC+1) * sizeof(int);
    // (*index) += 1;                            

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
     int global_warp_id = global_id >> 5; //global_id / WARP_SIZE;

    if (global_warp_id >= numblkC) return;
    global_warp_id = d_blkid[global_warp_id];

    const int nnzcstart = d_nnzb_C[global_warp_id];
    const int blknnzctotal = d_nnzb_C[global_warp_id+1] - nnzcstart;
    if (!blknnzctotal) return;

    const int local_warp_id = threadIdx.x >> 5; //threadIdx.x / WARP_SIZE;
    __shared__ unsigned char s_blkcsr_Idx_C[WARP_PER_BLOCK * SMEM_SIZE];
    __shared__ MAT_VAL_TYPE s_blkcsr_Val_C[WARP_PER_BLOCK * SMEM_SIZE];
    __shared__ unsigned char s_blkcsr_Ptr_C[WARP_PER_BLOCK * BLOCK_SIZE];
    unsigned char *s_blkcsr_Idx_C_local = &s_blkcsr_Idx_C[local_warp_id * SMEM_SIZE];
    MAT_VAL_TYPE *s_blkcsr_Val_C_local = &s_blkcsr_Val_C[local_warp_id * SMEM_SIZE];
    unsigned char *s_blkcsr_Ptr_C_local = &s_blkcsr_Ptr_C[local_warp_id * BLOCK_SIZE];


    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int sublane_id = lane_id % (WARP_SIZE / BLOCK_SIZE);
    const int subwarp_id = lane_id / (WARP_SIZE / BLOCK_SIZE);

  //  __shared__ unsigned int s_maskc[WARP_PER_BLOCK * BLOCK_SIZE];
   // unsigned int *s_maskc_local = &s_maskc[local_warp_id * BLOCK_SIZE];

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
/*
    const int blki = d_blkrowidxC[global_warp_id];
    const int blkj = d_blkcolidxC[global_warp_id];

    const int abase = d_blkrowptrA[blki];
    const int astop = d_blkrowptrA[blki+1];
    const int astart = d_blkcolidxA[abase];
    //const int aend = d_blkcolidxA[astop-1];
    int lena = astop - abase;

    const int bbase = d_blkcolptrB[blkj];
    const int bstop = d_blkcolptrB[blkj+1];
    const int bstart = d_blkrowidxB[bbase];
    //const int bend = d_blkrowidxB[bstop-1];
    int lenb = bstop - bbase;

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
*/
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



template <int SMEM_SIZE1, int SMEM_SIZE2>
__global__
void stir_spgemm_step4_cuda_kernel_smem_v2_scalar
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

}

template <int SMEM_SIZE1, int SMEM_SIZE2>
__global__
void stir_spgemm_step4_cuda_kernel_smem_v2
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
     int global_warp_id = global_id >> 5;//global_id / WARP_SIZE;

    if (global_warp_id >= numblkC) return;
    global_warp_id = d_blkid[global_warp_id];

    const int nnzcstart = d_nnzb_C[global_warp_id];
    const int blknnzctotal = d_nnzb_C[global_warp_id+1] - nnzcstart;
    if (!blknnzctotal) return;

    const int local_warp_id = threadIdx.x >> 5;//threadIdx.x / WARP_SIZE;
    __shared__ unsigned char s_blkcsr_Idx_C[WARP_PER_BLOCK * SMEM_SIZE1];
    __shared__ MAT_VAL_TYPE s_blkcsr_Val_C[WARP_PER_BLOCK * SMEM_SIZE1];
    __shared__ unsigned char s_blkcsr_Ptr_C[WARP_PER_BLOCK * BLOCK_SIZE];
    unsigned char *s_blkcsr_Idx_C_local = &s_blkcsr_Idx_C[local_warp_id * SMEM_SIZE1];
    MAT_VAL_TYPE *s_blkcsr_Val_C_local = &s_blkcsr_Val_C[local_warp_id * SMEM_SIZE1];
    unsigned char *s_blkcsr_Ptr_C_local = &s_blkcsr_Ptr_C[local_warp_id * BLOCK_SIZE];
    

    __shared__ unsigned char s_csrRowPtrB[WARP_PER_BLOCK * BLOCK_SIZE];
    unsigned char *s_csrRowPtrB_local = &s_csrRowPtrB[local_warp_id * BLOCK_SIZE];
    

    __shared__ int s_matched_posa[WARP_PER_BLOCK * SMEM_SIZE2];
    __shared__ int s_matched_posb[WARP_PER_BLOCK * SMEM_SIZE2];
    __shared__ int s_matchedcnt[WARP_PER_BLOCK];
    
    int *s_matched_posa_local = &s_matched_posa[local_warp_id * SMEM_SIZE2];
    int *s_matched_posb_local = &s_matched_posb[local_warp_id * SMEM_SIZE2];
    int *s_matchedcnt_local = &s_matchedcnt[local_warp_id];



    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int sublane_id = lane_id % (WARP_SIZE / BLOCK_SIZE);
    const int subwarp_id = lane_id / (WARP_SIZE / BLOCK_SIZE);

   // __shared__ unsigned int s_maskc[WARP_PER_BLOCK * BLOCK_SIZE];
   // unsigned int *s_maskc_local = &s_maskc[local_warp_id * BLOCK_SIZE];

    for (int i = lane_id; i < SMEM_SIZE1; i+=WARP_SIZE)
        s_blkcsr_Val_C_local[i] = 0;
    if (lane_id < BLOCK_SIZE)
        s_blkcsr_Ptr_C_local[lane_id] = d_blkcsr_Ptr_C[global_warp_id * BLOCK_SIZE + lane_id];
    //__syncthreads();

    unsigned int maskc = lane_id < BLOCK_SIZE ? d_blkmaskC[global_warp_id * BLOCK_SIZE + lane_id] : 0;//s_maskc_local[lane_id];
    unsigned char blknnzcstart = lane_id < BLOCK_SIZE ? s_blkcsr_Ptr_C_local[lane_id] : 0;
    if (!lane_id) 
        s_matchedcnt_local[0] = 0;
        
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
    int lena = astop - abase;

    const int bbase = d_blkcolptrB[blkj];
    const int bstop = d_blkcolptrB[blkj+1];
    int lenb = bstop - bbase;
    
    
    if (lena < lenb)
    {
        for (int i = lane_id; i < lena; i+= WARP_SIZE)
        {
            int idxa = d_blkcolidxA[abase+i];
            int res = binary_search_exact_kernel(&d_blkrowidxB[bbase], 0, lenb-1, idxa);
            if (res != -1)
            {
                int pos = atomicAdd(s_matchedcnt_local, 1);
                if (pos < SMEM_SIZE2)
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
                if (pos < SMEM_SIZE2)
                {
                    s_matched_posa_local[pos] = res;
                    s_matched_posb_local[pos] = i;
                }
            }
        }
    }

    int matchedcnt = s_matchedcnt_local[0];
    
    
    if (matchedcnt <= SMEM_SIZE2)
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
                    //atomicAdd(&s_blkcsr_Val_C_local[colidx * BLOCK_SIZE + rowidxa], val * valb);
                    
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

/*
            // sub-warp method (process row/num_subwarp one time, each row uses THREADS_PER_ROW threads)
            //const int THREADS_PER_ROW = 4;
            const int num_subwarp = WARP_SIZE / THREADS_PER_ROW;
            const int sublane_id = lane_id % THREADS_PER_ROW;
            const int subwarp_id = lane_id / THREADS_PER_ROW;

            const int nnzastart = d_nnzb_A[(abase+posa)];
            int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;

            if (lane_id < BLOCK_SIZE)
                s_csrRowPtrB_local[lane_id] = d_blkcsr_Ptr_B[(bbase+posb)*BLOCK_SIZE+lane_id];
            const int nnzbstart = d_nnzb_B[(bbase+posb)];
            int nnztotalb = d_nnzb_B[(bbase+posb) + 1] - nnzbstart;

            for (int rowidxa = subwarp_id; rowidxa < BLOCK_SIZE; rowidxa += num_subwarp)
            {
                int blkstarta = d_blkcsr_Ptr_A[(abase+posa)*BLOCK_SIZE+rowidxa];
                int blkstopa = rowidxa == BLOCK_SIZE - 1 ? nnztotala : d_blkcsr_Ptr_A[(abase+posa)*BLOCK_SIZE+rowidxa+1];

                int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];
                //int blkoffseta_stop = rowidxa == BLOCK_SIZE - 1 ? blknnzctotal : s_blkcsr_Ptr_C_local[rowidxa+1];

                for (int j = blkstarta; j < blkstopa; j++)
                {
                    unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+j];
                    //int rowidxa = lane_id; //rowcolidx >> 4;
                    int rowidxb = rowcolidx & 0xf;
                    MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart+j];

                    const int startb = s_csrRowPtrB_local[rowidxb]; //d_csrRowPtrB[rowidxb];
                    const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : s_csrRowPtrB_local[rowidxb+1]; //d_csrRowPtrB[rowidxb+1];

                    for (int k = startb + sublane_id; k < stopb; k+=THREADS_PER_ROW)
                    {
                        unsigned char colidx = d_blkcsr_Col_B[nnzbstart+k];
                        MAT_VAL_TYPE valb = d_blkcsr_Val_B[nnzbstart+k];

                    int cnt = 0;
                    //unsigned char colidx_exist = s_blkcsr_Idx_C_local[blknnzcstart];
                    while (colidx != s_blkcsr_Idx_C_local[blkoffseta + cnt])
                    {
                        cnt++;
                        //colidx_exist = s_blkcsr_Idx_C_local[blknnzcstart + cnt];
                    }
                    s_blkcsr_Val_C_local[blkoffseta + cnt] += val * valb;

//                        int cnt = binary_search_exact_uchar_kernel(&s_blkcsr_Idx_C_local[blkoffseta], 0, blkoffseta_stop-blkoffseta-1, colidx);
//                        if (cnt != -1) 
//                            s_blkcsr_Val_C_local[blkoffseta + cnt] += val * valb;
                            //atomicAdd(&s_blkcsr_Val_C_local[blkoffseta + cnt], val * valb);
                    }
                }
            }
*/
        }
    }
    else
    {
        const int astart = d_blkcolidxA[abase];
        const int aend = d_blkcolidxA[astop-1];
        const int bstart = d_blkrowidxB[bbase];
        const int bend = d_blkrowidxB[bstop-1];

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
    //__syncthreads();
    }


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


template <int SMEM_SIZE1, int SMEM_SIZE2, int THREADS_PER_ROW>
__global__
void stir_spgemm_step4_cuda_kernel_smem_v3
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
                            const MAT_VAL_TYPE * __restrict__ d_blkcsr_Val_B,
                            const unsigned char * __restrict__ d_blkcsr_Col_B,
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
     int global_warp_id = global_id >> 5;//global_id / WARP_SIZE;

    if (global_warp_id >= numblkC) return;
    global_warp_id = d_blkid[global_warp_id];

    const int nnzcstart = d_nnzb_C[global_warp_id];
    const int blknnzctotal = d_nnzb_C[global_warp_id+1] - nnzcstart;
    if (!blknnzctotal) return;

    const int local_warp_id = threadIdx.x >> 5;//threadIdx.x / WARP_SIZE;
    __shared__ unsigned char s_blkcsr_Idx_C[WARP_PER_BLOCK * SMEM_SIZE1];
    __shared__ MAT_VAL_TYPE s_blkcsr_Val_C[WARP_PER_BLOCK * SMEM_SIZE1];
    __shared__ unsigned char s_blkcsr_Ptr_C[WARP_PER_BLOCK * BLOCK_SIZE];
    unsigned char *s_blkcsr_Idx_C_local = &s_blkcsr_Idx_C[local_warp_id * SMEM_SIZE1];
    MAT_VAL_TYPE *s_blkcsr_Val_C_local = &s_blkcsr_Val_C[local_warp_id * SMEM_SIZE1];
    unsigned char *s_blkcsr_Ptr_C_local = &s_blkcsr_Ptr_C[local_warp_id * BLOCK_SIZE];

    __shared__ unsigned char s_csrRowPtrB[WARP_PER_BLOCK * BLOCK_SIZE];
    unsigned char *s_csrRowPtrB_local = &s_csrRowPtrB[local_warp_id * BLOCK_SIZE];
    
    __shared__ int s_matched_posa[WARP_PER_BLOCK * SMEM_SIZE2];
    __shared__ int s_matched_posb[WARP_PER_BLOCK * SMEM_SIZE2];
    __shared__ int s_matchedcnt[WARP_PER_BLOCK];
    
    int *s_matched_posa_local = &s_matched_posa[local_warp_id * SMEM_SIZE2];
    int *s_matched_posb_local = &s_matched_posb[local_warp_id * SMEM_SIZE2];
    int *s_matchedcnt_local = &s_matchedcnt[local_warp_id];

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;


    //__shared__ unsigned int s_maskc[WARP_PER_BLOCK * BLOCK_SIZE];
    //unsigned int *s_maskc_local = &s_maskc[local_warp_id * BLOCK_SIZE];

    for (int i = lane_id; i < SMEM_SIZE1; i+=WARP_SIZE)
       s_blkcsr_Val_C_local[i] = 0.0;
    if (lane_id < BLOCK_SIZE)
        s_blkcsr_Ptr_C_local[lane_id] = d_blkcsr_Ptr_C[global_warp_id * BLOCK_SIZE + lane_id];
    //__syncthreads();

    unsigned int maskc = lane_id < BLOCK_SIZE ? d_blkmaskC[global_warp_id * BLOCK_SIZE + lane_id] : 0;//s_maskc_local[lane_id];
    unsigned char blknnzcstart = lane_id < BLOCK_SIZE ? s_blkcsr_Ptr_C_local[lane_id] : 0;
    if (!lane_id) 
        s_matchedcnt_local[0] = 0;
        
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
    int lena = astop - abase;

    const int bbase = d_blkcolptrB[blkj];
    const int bstop = d_blkcolptrB[blkj+1];
    int lenb = bstop - bbase;
    
    if (lena < lenb)
    {
        for (int i = lane_id; i < lena; i+= WARP_SIZE)
        {
            int idxa = d_blkcolidxA[abase+i];
            int res = binary_search_exact_kernel(&d_blkrowidxB[bbase], 0, lenb-1, idxa);
            if (res != -1)
            {
                int pos = atomicAdd(s_matchedcnt_local, 1);
                if (pos < SMEM_SIZE2)
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
                if (pos < SMEM_SIZE2)
                {
                    s_matched_posa_local[pos] = res;
                    s_matched_posb_local[pos] = i;
                }
            }
        }
    }

    int matchedcnt = s_matchedcnt_local[0];
    
    
    if (matchedcnt <= SMEM_SIZE2)
    {
        for (int posi = 0; posi < matchedcnt; posi++)
        {
            int posa = s_matched_posa_local[posi];
            int posb = s_matched_posb_local[posi];
/*
            // nonatomic method
            if (lane_id < BLOCK_SIZE)
            {
                const int nnzastart = d_nnzb_A[(abase+posa)];
                int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;
                int blkstarta = d_blkcsr_Ptr_A[(abase+posa)*BLOCK_SIZE+lane_id];
                int blkstopa = __shfl_down_sync(0xffffffff, blkstarta, 1);
                blkstopa = lane_id == BLOCK_SIZE - 1 ? nnztotala : blkstopa;

                const int nnzbstart = d_nnzb_B[(bbase+posb)];
                int nnztotalb = d_nnzb_B[(bbase+posb) + 1] - nnzbstart;
                s_csrRowPtrB_local[lane_id] = d_blkcsr_Ptr_B[(bbase+posb)*BLOCK_SIZE+lane_id];

                int blkoffseta = s_blkcsr_Ptr_C_local[lane_id];
                int blkoffseta_stop = __shfl_down_sync(0xffffffff, blkoffseta, 1);
                blkoffseta_stop = lane_id == BLOCK_SIZE - 1 ? blknnzctotal : blkoffseta_stop;

                for (int j = blkstarta; j < blkstopa; j++)
                {
                    unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+j];
                    //int rowidxa = lane_id; //rowcolidx >> 4;
                    int rowidxb = rowcolidx & 0xf;
                    MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart+j];

                    const int startb = s_csrRowPtrB_local[rowidxb]; //d_csrRowPtrB[rowidxb];
                    const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : s_csrRowPtrB_local[rowidxb+1]; //d_csrRowPtrB[rowidxb+1];

                    for (int k = startb; k < stopb; k++)
                    {
                        unsigned char colidx = d_blkcsr_Col_B[nnzbstart+k];
                        MAT_VAL_TYPE valb = d_blkcsr_Val_B[nnzbstart+k];

                    int cnt = 0;
                    //unsigned char colidx_exist = s_blkcsr_Idx_C_local[blknnzcstart];
                    while (colidx != s_blkcsr_Idx_C_local[blkoffseta + cnt])
                    {
                        cnt++;
                        //colidx_exist = s_blkcsr_Idx_C_local[blknnzcstart + cnt];
                    }
                    s_blkcsr_Val_C_local[blkoffseta + cnt] += val * valb;

//                        int cnt = binary_search_exact_uchar_kernel(&s_blkcsr_Idx_C_local[blkoffseta], 0, blkoffseta_stop-blkoffseta-1, colidx);
//                        if (cnt != -1) 
//                            s_blkcsr_Val_C_local[blkoffseta + cnt] += val * valb;
                            //atomicAdd(&s_blkcsr_Val_C_local[blkoffseta + cnt], val * valb);
                    }
                }
            }


            // sub-warp method (process row/num_subwarp one time, each row uses THREADS_PER_ROW threads)
            //const int THREADS_PER_ROW = 4;
            const int num_subwarp = WARP_SIZE / THREADS_PER_ROW;
            const int sublane_id = lane_id % THREADS_PER_ROW;
            const int subwarp_id = lane_id / THREADS_PER_ROW;

            const int nnzastart = d_nnzb_A[(abase+posa)];
            int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;

            if (lane_id < BLOCK_SIZE)
                s_csrRowPtrB_local[lane_id] = d_blkcsr_Ptr_B[(bbase+posb)*BLOCK_SIZE+lane_id];
            const int nnzbstart = d_nnzb_B[(bbase+posb)];
            int nnztotalb = d_nnzb_B[(bbase+posb) + 1] - nnzbstart;

            for (int rowidxa = subwarp_id; rowidxa < BLOCK_SIZE; rowidxa += num_subwarp)
            {
                int blkstarta = d_blkcsr_Ptr_A[(abase+posa)*BLOCK_SIZE+rowidxa];
                int blkstopa = rowidxa == BLOCK_SIZE - 1 ? nnztotala : d_blkcsr_Ptr_A[(abase+posa)*BLOCK_SIZE+rowidxa+1];

                int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];
                //int blkoffseta_stop = rowidxa == BLOCK_SIZE - 1 ? blknnzctotal : s_blkcsr_Ptr_C_local[rowidxa+1];

                for (int j = blkstarta; j < blkstopa; j++)
                {
                    unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+j];
                    //int rowidxa = lane_id; //rowcolidx >> 4;
                    int rowidxb = rowcolidx & 0xf;
                    MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart+j];

                    const int startb = s_csrRowPtrB_local[rowidxb]; //d_csrRowPtrB[rowidxb];
                    const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : s_csrRowPtrB_local[rowidxb+1]; //d_csrRowPtrB[rowidxb+1];

                    for (int k = startb + sublane_id; k < stopb; k+=THREADS_PER_ROW)
                    {
                        unsigned char colidx = d_blkcsr_Col_B[nnzbstart+k];
                        MAT_VAL_TYPE valb = d_blkcsr_Val_B[nnzbstart+k];

                    int cnt = 0;
                    //unsigned char colidx_exist = s_blkcsr_Idx_C_local[blknnzcstart];
                    while (colidx != s_blkcsr_Idx_C_local[blkoffseta + cnt])
                    {
                        cnt++;
                        //colidx_exist = s_blkcsr_Idx_C_local[blknnzcstart + cnt];
                    }
                    s_blkcsr_Val_C_local[blkoffseta + cnt] += val * valb;

//                        int cnt = binary_search_exact_uchar_kernel(&s_blkcsr_Idx_C_local[blkoffseta], 0, blkoffseta_stop-blkoffseta-1, colidx);
//                        if (cnt != -1) 
//                            s_blkcsr_Val_C_local[blkoffseta + cnt] += val * valb;
                            //atomicAdd(&s_blkcsr_Val_C_local[blkoffseta + cnt], val * valb);
                    }
                }
            }
*/

            // atomic method
            const int nnzastart = d_nnzb_A[(abase+posa)];
            int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;
            //unsigned char *d_csrRowPtrB = &d_blkcsr_Ptr_B[(bbase+posb)*BLOCK_SIZE];
            if (lane_id < BLOCK_SIZE)
                s_csrRowPtrB_local[lane_id] = d_blkcsr_Ptr_B[(bbase+posb)*BLOCK_SIZE+lane_id];
            const int nnzbstart = d_nnzb_B[(bbase+posb)];
            int nnztotalb = d_nnzb_B[(bbase+posb) + 1] - nnzbstart;

            for (int i = lane_id; i < nnztotala; i+= WARP_SIZE)
            {
                unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+i];
                int rowidxa = rowcolidx >> 4;
                int rowidxb = rowcolidx & 0xf;
                MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart+i];
                int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];
                int blkoffseta_stop = rowidxa == BLOCK_SIZE - 1 ? blknnzctotal : s_blkcsr_Ptr_C_local[rowidxa+1];
                
                const int startb = s_csrRowPtrB_local[rowidxb]; //d_csrRowPtrB[rowidxb];
                const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : s_csrRowPtrB_local[rowidxb+1]; //d_csrRowPtrB[rowidxb+1];
                for (int k = startb; k < stopb; k++)
                {
                    //unsigned char colidx_packet = d_csrColIdxB[k/2];
                    //unsigned char colidx = (k % 2 == 0) ? (colidx >> 4 & 0x0F) : (colidx & 0x0F);
                    unsigned char colidx = ld_gbl_uchar(&d_blkcsr_Col_B[nnzbstart+k]);
                    MAT_VAL_TYPE valb = ld_gbl_real(&d_blkcsr_Val_B[nnzbstart+k]);
                    //maskc = maskc | (0x1 << (BLOCK_SIZE - colidx - 1));
                    //atomicOr(&s_maskc_local[rowidxa], (unsigned int)(0x1 << (BLOCK_SIZE - colidx - 1)));
                    //s_dnsidx_local[subwarp_id * BLOCK_SIZE + colidx] = 1;
                    //s_blkcsr_Val_C_local[subwarp_id * BLOCK_SIZE + colidx] += val * valb;
                    //atomicAdd(&s_blkcsr_Val_C_local[rowidxa * BLOCK_SIZE + colidx], val * valb);
                    //atomicAdd(&s_blkcsr_Val_C_local[colidx * BLOCK_SIZE + rowidxa], val * valb);

                    //unsigned char colidx_exist = s_blkcsr_Idx_C_local[blknnzcstart];            
/*
                    int cnt = 0;
                    while (colidx != s_blkcsr_Idx_C_local[blkoffseta + cnt])
                    {
                        cnt++;
                        //colidx_exist = s_blkcsr_Idx_C_local[blknnzcstart + cnt];
                    }
                    //s_blkcsr_Val_C_local[0] = cnt * val;
                    atomicAdd(&s_blkcsr_Val_C_local[blkoffseta + cnt], val * valb);
*/
                    int cnt = binary_search_exact_uchar_kernel(&s_blkcsr_Idx_C_local[blkoffseta], 0, blkoffseta_stop-blkoffseta-1, colidx);
                    if (cnt != -1) 
                        //s_blkcsr_Val_C_local[blkoffseta + cnt] += val * valb;
                        atomicAdd(&s_blkcsr_Val_C_local[blkoffseta + cnt], val * valb);
                    //atomicAdd(&s_blkcsr_Val_C_local[colidx * BLOCK_SIZE + rowidxa], val * valb);
                }
            }
        }
    }
    else
    {
        const int astart = d_blkcolidxA[abase];
        const int aend = d_blkcolidxA[astop-1];
        const int bstart = d_blkrowidxB[bbase];
        const int bend = d_blkrowidxB[bstop-1];

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

/*
                // sub-warp method (process row/num_subwarp one time, each row uses THREADS_PER_ROW threads)
                //const int THREADS_PER_ROW = 4;
                const int num_subwarp = WARP_SIZE / THREADS_PER_ROW;
                const int sublane_id = lane_id % THREADS_PER_ROW;
                const int subwarp_id = lane_id / THREADS_PER_ROW;

                const int nnzastart = d_nnzb_A[(abase+posa)];
                int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;

                if (lane_id < BLOCK_SIZE)
                    s_csrRowPtrB_local[lane_id] = d_blkcsr_Ptr_B[(bbase+posb)*BLOCK_SIZE+lane_id];
                const int nnzbstart = d_nnzb_B[(bbase+posb)];
                int nnztotalb = d_nnzb_B[(bbase+posb) + 1] - nnzbstart;

                for (int rowidxa = subwarp_id; rowidxa < BLOCK_SIZE; rowidxa += num_subwarp)
                {
                    int blkstarta = d_blkcsr_Ptr_A[(abase+posa)*BLOCK_SIZE+rowidxa];
                    int blkstopa = rowidxa == BLOCK_SIZE - 1 ? nnztotala : d_blkcsr_Ptr_A[(abase+posa)*BLOCK_SIZE+rowidxa+1];

                    int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];
                    //int blkoffseta_stop = rowidxa == BLOCK_SIZE - 1 ? blknnzctotal : s_blkcsr_Ptr_C_local[rowidxa+1];

                    for (int j = blkstarta; j < blkstopa; j++)
                    {
                        unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+j];
                        //int rowidxa = lane_id; //rowcolidx >> 4;
                        int rowidxb = rowcolidx & 0xf;
                        MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart+j];

                        const int startb = s_csrRowPtrB_local[rowidxb]; //d_csrRowPtrB[rowidxb];
                        const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : s_csrRowPtrB_local[rowidxb+1]; //d_csrRowPtrB[rowidxb+1];

                        for (int k = startb + sublane_id; k < stopb; k+=THREADS_PER_ROW)
                        {
                            unsigned char colidx = d_blkcsr_Col_B[nnzbstart+k];
                            MAT_VAL_TYPE valb = d_blkcsr_Val_B[nnzbstart+k];

                        int cnt = 0;
                        //unsigned char colidx_exist = s_blkcsr_Idx_C_local[blknnzcstart];
                        while (colidx != s_blkcsr_Idx_C_local[blkoffseta + cnt])
                        {
                            cnt++;
                            //colidx_exist = s_blkcsr_Idx_C_local[blknnzcstart + cnt];
                        }
                        s_blkcsr_Val_C_local[blkoffseta + cnt] += val * valb;

    //                        int cnt = binary_search_exact_uchar_kernel(&s_blkcsr_Idx_C_local[blkoffseta], 0, blkoffseta_stop-blkoffseta-1, colidx);
    //                        if (cnt != -1) 
    //                            s_blkcsr_Val_C_local[blkoffseta + cnt] += val * valb;
                                //atomicAdd(&s_blkcsr_Val_C_local[blkoffseta + cnt], val * valb);
                        }
                    }
                }
*/

                const int nnzastart = d_nnzb_A[(abase+posa)];
                int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;
                //unsigned char *d_csrRowPtrB = &d_blkcsr_Ptr_B[(bbase+posb)*BLOCK_SIZE];
                if (lane_id < BLOCK_SIZE)
                    s_csrRowPtrB_local[lane_id] = d_blkcsr_Ptr_B[(bbase+posb)*BLOCK_SIZE+lane_id];
                const int nnzbstart = d_nnzb_B[(bbase+posb)];
                int nnztotalb = d_nnzb_B[(bbase+posb) + 1] - nnzbstart;

                for (int i = lane_id; i < nnztotala; i+= WARP_SIZE)
                {
                    unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+i];
                    int rowidxa = rowcolidx >> 4;
                    int rowidxb = rowcolidx & 0xf;
                    MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart+i];
                    int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];

                    const int startb = s_csrRowPtrB_local[rowidxb]; //d_csrRowPtrB[rowidxb];
                    const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : s_csrRowPtrB_local[rowidxb+1]; //d_csrRowPtrB[rowidxb+1];
                    for (int k = startb; k < stopb; k++)
                    {
                        //unsigned char colidx_packet = d_csrColIdxB[k/2];
                        //unsigned char colidx = (k % 2 == 0) ? (colidx >> 4 & 0x0F) : (colidx & 0x0F);
                        unsigned char colidx = ld_gbl_uchar(&d_blkcsr_Col_B[nnzbstart+k]);
                        MAT_VAL_TYPE valb = ld_gbl_real(&d_blkcsr_Val_B[nnzbstart+k]);
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
                        //s_blkcsr_Val_C_local[blkoffseta + cnt] += val * valb;
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
    //__syncthreads();
    }

    for (int i = lane_id; i < blknnzctotal; i+= WARP_SIZE)
    {
        d_blkcsr_Col_C[nnzcstart + i] = s_blkcsr_Idx_C_local[i];
        d_blkcsr_Val_C[nnzcstart + i] = s_blkcsr_Val_C_local[i];
    }
}

template <int SMEM_SIZE, int THREADS_PER_ROW>
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
     int global_warp_id = global_id >> 5;//global_id / WARP_SIZE;

    if (global_warp_id >= numblkC) return;
    global_warp_id = d_blkid[global_warp_id];

    const int nnzcstart = d_nnzb_C[global_warp_id];
    const int blknnzctotal = d_nnzb_C[global_warp_id+1] - nnzcstart;
    if (!blknnzctotal) return;

    const int local_warp_id = threadIdx.x >> 5; //threadIdx.x / WARP_SIZE;
    __shared__ MAT_VAL_TYPE s_blkcsr_Val_C[WARP_PER_BLOCK * BLOCK_SIZE * BLOCK_SIZE];
    MAT_VAL_TYPE *s_blkcsr_Val_C_local = &s_blkcsr_Val_C[local_warp_id * BLOCK_SIZE * BLOCK_SIZE];

    __shared__ int s_matched_posa[WARP_PER_BLOCK * SMEM_SIZE];
    __shared__ int s_matched_posb[WARP_PER_BLOCK * SMEM_SIZE];
    __shared__ int s_matchedcnt[WARP_PER_BLOCK];

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int sublane_id = lane_id % (WARP_SIZE / BLOCK_SIZE);
    const int subwarp_id = lane_id / (WARP_SIZE / BLOCK_SIZE);
    //unsigned int maskc = 0;
    __shared__ unsigned int s_maskc[WARP_PER_BLOCK * BLOCK_SIZE];
    unsigned int *s_maskc_local = &s_maskc[local_warp_id * BLOCK_SIZE];

    int *s_matched_posa_local = &s_matched_posa[local_warp_id * SMEM_SIZE];
    int *s_matched_posb_local = &s_matched_posb[local_warp_id * SMEM_SIZE];
    int *s_matchedcnt_local = &s_matchedcnt[local_warp_id];


    //#pragma unroll
    //for (int i = 0; i < BLOCK_SIZE; i++)
        //if (lane_id < BLOCK_SIZE)
    //        s_blkcsr_Val_C_local[i * BLOCK_SIZE + lane_id] = 0;
            
    s_blkcsr_Val_C_local[lane_id] = 0.0;
    s_blkcsr_Val_C_local[32 + lane_id] = 0.0;
    s_blkcsr_Val_C_local[64 + lane_id] = 0.0;
    s_blkcsr_Val_C_local[96 + lane_id] = 0.0;
    s_blkcsr_Val_C_local[128 + lane_id] = 0.0;
    s_blkcsr_Val_C_local[160 + lane_id] = 0.0;
    s_blkcsr_Val_C_local[192 + lane_id] = 0.0;
    s_blkcsr_Val_C_local[224 + lane_id] = 0.0;
    
    if (!lane_id) 
        s_matchedcnt_local[0] = 0;

    //if (lane_id < BLOCK_SIZE)
    //    s_maskc_local[lane_id] = d_blkmaskC[global_warp_id * BLOCK_SIZE + lane_id];
    //__syncthreads();

    const int blki = d_blkrowidxC[global_warp_id];
    const int blkj = d_blkcolidxC[global_warp_id];

    const int abase = d_blkrowptrA[blki];
    const int astop = d_blkrowptrA[blki+1];
    int lena = d_blkrowptrA[blki+1] - abase;

    const int bbase = d_blkcolptrB[blkj];
    const int bstop = d_blkcolptrB[blkj+1];
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

    /*        // sub-warp method (process row/num_subwarp one time, each row uses THREADS_PER_ROW threads)
            //const int THREADS_PER_ROW = 4;
            const int num_subwarp = WARP_SIZE / THREADS_PER_ROW;
            const int sublane_id = lane_id % THREADS_PER_ROW;
            const int subwarp_id = lane_id / THREADS_PER_ROW;

            const int nnzastart = d_nnzb_A[(abase+posa)];
            int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;

            //if (lane_id < BLOCK_SIZE)
            //    s_csrRowPtrB_local[lane_id] = d_blkcsr_Ptr_B[(bbase+posb)*BLOCK_SIZE+lane_id];
            unsigned char *d_csrRowPtrB = &d_blkcsr_Ptr_B[(bbase+posb)*BLOCK_SIZE];
            const int nnzbstart = d_nnzb_B[(bbase+posb)];
            int nnztotalb = d_nnzb_B[(bbase+posb) + 1] - nnzbstart;

            for (int rowidxa = subwarp_id; rowidxa < BLOCK_SIZE; rowidxa += num_subwarp)
            {
                int blkstarta = d_blkcsr_Ptr_A[(abase+posa)*BLOCK_SIZE+rowidxa];
                int blkstopa = rowidxa == BLOCK_SIZE - 1 ? nnztotala : d_blkcsr_Ptr_A[(abase+posa)*BLOCK_SIZE+rowidxa+1];

                //int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];
                //int blkoffseta_stop = rowidxa == BLOCK_SIZE - 1 ? blknnzctotal : s_blkcsr_Ptr_C_local[rowidxa+1];

                for (int j = blkstarta; j < blkstopa; j++)
                {
                    unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+j];
                    //int rowidxa = lane_id; //rowcolidx >> 4;
                    int rowidxb = rowcolidx & 0xf;
                    MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart+j];

                    const int startb = d_csrRowPtrB[rowidxb];
                    const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : d_csrRowPtrB[rowidxb+1];

                    for (int k = startb + sublane_id; k < stopb; k+=THREADS_PER_ROW)
                    {
                        unsigned char colidx = d_blkcsr_Col_B[nnzbstart+k];
                        MAT_VAL_TYPE valb = d_blkcsr_Val_B[nnzbstart+k];
                        s_blkcsr_Val_C_local[rowidxa * BLOCK_SIZE + colidx] += val * valb;

//                        int cnt = binary_search_exact_uchar_kernel(&s_blkcsr_Idx_C_local[blkoffseta], 0, blkoffseta_stop-blkoffseta-1, colidx);
//                        if (cnt != -1) 
//                            s_blkcsr_Val_C_local[blkoffseta + cnt] += val * valb;
                            //atomicAdd(&s_blkcsr_Val_C_local[blkoffseta + cnt], val * valb);
                    }
                }
            }
*/
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
                    //s_blkcsr_Val_C_local[rowidxa * BLOCK_SIZE + colidx] += val * valb;
                    //atomicAdd(&s_blkcsr_Val_C_local[colidx * BLOCK_SIZE + rowidxa], val * valb);
                }
            }

        }
    }
    else 
    {
        const int astart = d_blkcolidxA[abase];
        const int aend = d_blkcolidxA[astop-1];
        const int bstart = d_blkrowidxB[bbase];
        const int bend = d_blkrowidxB[bstop-1];

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
    //{
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
    
    //}
    //else
    //{

            // sub-warp method (process row/num_subwarp one time, each row uses THREADS_PER_ROW threads)
            //const int THREADS_PER_ROW = 4;
            const int num_subwarp = WARP_SIZE / THREADS_PER_ROW;
            const int sublane_id = lane_id % THREADS_PER_ROW;
            const int subwarp_id = lane_id / THREADS_PER_ROW;

            const int nnzastart = d_nnzb_A[(abase+posa)];
            int nnztotala = d_nnzb_A[(abase+posa) + 1] - nnzastart;

            //if (lane_id < BLOCK_SIZE)
            //    s_csrRowPtrB_local[lane_id] = d_blkcsr_Ptr_B[(bbase+posb)*BLOCK_SIZE+lane_id];
            unsigned char *d_csrRowPtrB = &d_blkcsr_Ptr_B[(bbase+posb)*BLOCK_SIZE];
            const int nnzbstart = d_nnzb_B[(bbase+posb)];
            int nnztotalb = d_nnzb_B[(bbase+posb) + 1] - nnzbstart;

            for (int rowidxa = subwarp_id; rowidxa < BLOCK_SIZE; rowidxa += num_subwarp)
            {
                int blkstarta = d_blkcsr_Ptr_A[(abase+posa)*BLOCK_SIZE+rowidxa];
                int blkstopa = rowidxa == BLOCK_SIZE - 1 ? nnztotala : d_blkcsr_Ptr_A[(abase+posa)*BLOCK_SIZE+rowidxa+1];

                //int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];
                //int blkoffseta_stop = rowidxa == BLOCK_SIZE - 1 ? blknnzctotal : s_blkcsr_Ptr_C_local[rowidxa+1];

                for (int j = blkstarta; j < blkstopa; j++)
                {
                    unsigned char rowcolidx = d_blkcsr_Col_A[nnzastart+j];
                    //int rowidxa = lane_id; //rowcolidx >> 4;
                    int rowidxb = rowcolidx & 0xf;
                    MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart+j];

                    const int startb = d_csrRowPtrB[rowidxb];
                    const int stopb = rowidxb == BLOCK_SIZE - 1 ? nnztotalb : d_csrRowPtrB[rowidxb+1];

                    for (int k = startb + sublane_id; k < stopb; k+=THREADS_PER_ROW)
                    {
                        unsigned char colidx = d_blkcsr_Col_B[nnzbstart+k];
                        MAT_VAL_TYPE valb = d_blkcsr_Val_B[nnzbstart+k];
                        s_blkcsr_Val_C_local[rowidxa * BLOCK_SIZE + colidx] += val * valb;

//                        int cnt = binary_search_exact_uchar_kernel(&s_blkcsr_Idx_C_local[blkoffseta], 0, blkoffseta_stop-blkoffseta-1, colidx);
//                        if (cnt != -1) 
//                            s_blkcsr_Val_C_local[blkoffseta + cnt] += val * valb;
                            //atomicAdd(&s_blkcsr_Val_C_local[blkoffseta + cnt], val * valb);
                    }
                }
            }
*/
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
                        //s_blkcsr_Val_C_local[rowidxa * BLOCK_SIZE + colidx] += val * valb;
                        //atomicAdd(&s_blkcsr_Val_C_local[colidx * BLOCK_SIZE + rowidxa], val * valb);
                    }
                }

    //}
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
int *d_blksmem_tny_cnt, int *d_blksmem_sml_cnt, int *d_blksmem_lrg_cnt, int *d_blksmem_dns_cnt, 
int *d_blkid_smem_tny, int *d_blkid_smem_sml, int *d_blkid_smem_lrg, int *d_blkid_smem_dns)
{
    // struct timeval tv;

    int *d_blkrowptrA;
    int *d_blkcolidxA; 
    int *d_nnzb_A; 
    MAT_VAL_TYPE *d_blkcsr_Val_A;
    unsigned char *d_blkcsr_Col_A;
    unsigned char *d_blkcsr_Ptr_A;

    cudaMalloc((void **)&d_blkrowptrA, (blkmA+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (blkmA+1) * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcolidxA, numblkA * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkA * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_nnzb_A, (numblkA+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (numblkA+1) * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcsr_Val_A, nnzA * sizeof(MAT_VAL_TYPE));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + nnzA * sizeof(MAT_VAL_TYPE);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcsr_Col_A, nnzA * sizeof(unsigned char));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + nnzA * sizeof(unsigned char);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcsr_Ptr_A, numblkA * BLOCK_SIZE  * sizeof(unsigned char));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkA * BLOCK_SIZE  * sizeof(unsigned char);
    // (*index) += 1;


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
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (blknB+1) * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkrowidxB, numblkB * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkB * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_nnzb_B, (numblkB+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (numblkB+1) * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcsr_Val_B, nnzB * sizeof(MAT_VAL_TYPE));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + nnzB * sizeof(MAT_VAL_TYPE);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcsr_Col_B, nnzB * sizeof(unsigned char));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + nnzB * sizeof(unsigned char);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcsr_Ptr_B, numblkB * BLOCK_SIZE  * sizeof(unsigned char));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkB * BLOCK_SIZE  * sizeof(unsigned char);
    // (*index) += 1;


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
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (numblkC+1) * sizeof(int);
    // (*index) += 1;

    //cudaMemcpy(d_blkrowptrC,     blkrowptrC,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemcpy(d_blkrowidxC,     blkrowidxC,     numblkC * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemcpy(d_blkcolidxC,     blkcolidxC,     numblkC * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemset(d_nnzb_C, 0, numblkC * sizeof(int));
cudaMemcpy(d_nnzb_C,     nnzb_C,     (numblkC+1) * sizeof(int),              cudaMemcpyHostToDevice);

    struct timeval t1, t2;

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

int blksmem_tny_cnt = 0;
int blksmem_sml_cnt = 0;
int blksmem_lrg_cnt = 0;
int blksmem_dns_cnt = 0;

cudaMemcpy(&blksmem_tny_cnt,     d_blksmem_tny_cnt,     sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(&blksmem_sml_cnt,     d_blksmem_sml_cnt,     sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(&blksmem_lrg_cnt,     d_blksmem_lrg_cnt,     sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(&blksmem_dns_cnt,     d_blksmem_dns_cnt,     sizeof(int), cudaMemcpyDeviceToHost);

    int num_threads = WARP_SIZE * WARP_PER_BLOCK;

    struct timeval tt1, tt2;
// tiny : 1 - 32
    cudaDeviceSynchronize();
    gettimeofday(&tt1, NULL);
    int num_blocks = ceil((double)blksmem_tny_cnt/(double)WARP_PER_BLOCK);
    stir_spgemm_step4_cuda_kernel_smem_v2<SMEM_TNY_TH,32><<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, 
d_blkcsr_Col_C,d_blkcsr_Val_C,
d_nnzb_C, d_blkmaskC, blksmem_tny_cnt, d_blkid_smem_tny);
    cudaDeviceSynchronize();
    gettimeofday(&tt2, NULL);
    double time41 = (tt2.tv_sec - tt1.tv_sec) * 1000.0 + (tt2.tv_usec - tt1.tv_usec) / 1000.0;
    printf("41 blksmem_tny_cnt = %i, time41 = %.2f ms\n", blksmem_tny_cnt, time41);

// small : 33 - 64
    cudaDeviceSynchronize();
    gettimeofday(&tt1, NULL);
     num_blocks = ceil((double)blksmem_sml_cnt/(double)WARP_PER_BLOCK);

    //cudaFuncSetAttribute(stir_spgemm_step4_cuda_kernel_smem_v3, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    //cudaFuncSetAttribute(stir_spgemm_step4_cuda_kernel_smem_v3<SMEM_SIZE1, SMEM_SIZE2, THREADS_PER_ROW>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    stir_spgemm_step4_cuda_kernel_smem_v3<SMEM_SML_TH,128,4><<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, 
d_blkcsr_Col_C,d_blkcsr_Val_C,
d_nnzb_C, d_blkmaskC, blksmem_sml_cnt, d_blkid_smem_sml);
    cudaDeviceSynchronize();
    gettimeofday(&tt2, NULL);
    double time42 = (tt2.tv_sec - tt1.tv_sec) * 1000.0 + (tt2.tv_usec - tt1.tv_usec) / 1000.0;
    printf("42 blksmem_sml_cnt = %i, time42 = %.2f ms\n", blksmem_sml_cnt, time42);

// large : 65 - 128
    cudaDeviceSynchronize();
    gettimeofday(&tt1, NULL);
     num_blocks = ceil((double)blksmem_lrg_cnt/(double)WARP_PER_BLOCK);
    stir_spgemm_step4_cuda_kernel_smem_v3<SMEM_LRG_TH,128,4><<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, 
d_blkcsr_Col_C,d_blkcsr_Val_C,
d_nnzb_C, d_blkmaskC, blksmem_lrg_cnt, d_blkid_smem_lrg);
    cudaDeviceSynchronize();
    gettimeofday(&tt2, NULL);
    double time43 = (tt2.tv_sec - tt1.tv_sec) * 1000.0 + (tt2.tv_usec - tt1.tv_usec) / 1000.0;
    printf("43 blksmem_lrg_cnt = %i, time43 = %.2f ms\n", blksmem_lrg_cnt, time43);
/*    num_blocks = ceil((double)blkdns_cnt/(double)WARP_PER_BLOCK);

    stir_spgemm_step4_cuda_kernel_dns<<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, 
d_blkcsr_Col_C,d_blkcsr_Val_C,
d_nnzb_C, blkdns_cnt, d_blkid_dns);
*/


// dns : 129 - dns
    cudaDeviceSynchronize();
    gettimeofday(&tt1, NULL);

     num_threads = WARP_SIZE * WARP_PER_BLOCK;
     num_blocks = ceil((double)blksmem_dns_cnt/(double)WARP_PER_BLOCK);
    stir_spgemm_step4_cuda_kernel_dns<128,8><<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, 
d_blkcsr_Col_C,d_blkcsr_Val_C,
d_nnzb_C, d_blkmaskC, blksmem_dns_cnt, d_blkid_smem_dns);


    cudaDeviceSynchronize();
    gettimeofday(&tt2, NULL);

    double time44 = (tt2.tv_sec - tt1.tv_sec) * 1000.0 + (tt2.tv_usec - tt1.tv_usec) / 1000.0;

    printf("44 blksmem_dns_cnt = %i, time44 = %.2f ms\n", blksmem_dns_cnt, time44);


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
    
    // for (int i=0 ; i < nnzC ; i++)
    // {
    //     printf("%f   ",h_blkcsr_Val_C[i]);
    // }
    // printf("\n");

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
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (blkmA+1) * sizeof(int);
    // (*index) += 1;                            

    cudaFree(d_blkcolidxA); 
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkA * sizeof(int);
    // (*index) += 1;                             

    cudaFree(d_nnzb_A); 
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (numblkA+1) * sizeof(int);
    // (*index) += 1;

    cudaFree(d_blkcsr_Val_A);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - nnzA * sizeof(MAT_VAL_TYPE);
    // (*index) += 1;                            

    cudaFree(d_blkcsr_Col_A);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - nnzA * sizeof(unsigned char);
    // (*index) += 1;

    cudaFree(d_blkcsr_Ptr_A);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkA * BLOCK_SIZE  * sizeof(unsigned char);
    // (*index) += 1;

    cudaFree(d_blkcolptrB);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (blknB+1) * sizeof(int);
    // (*index) += 1;                            

    cudaFree(d_blkrowidxB); 
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkB * sizeof(int);
    // (*index) += 1;                             

    cudaFree(d_nnzb_B); 
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (numblkB+1) * sizeof(int);
    // (*index) += 1;                            

    cudaFree(d_blkcsr_Val_B);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - nnzB * sizeof(MAT_VAL_TYPE);
    // (*index) += 1;                            


    cudaFree(d_blkcsr_Col_B);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - nnzB * sizeof(unsigned char);
    // (*index) += 1;                            

    cudaFree(d_blkcsr_Ptr_B);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkB * BLOCK_SIZE  * sizeof(unsigned char);
    // (*index) += 1;

    //cudaFree(d_blkrowptrC);
    //cudaFree(d_blkrowidxC);
    //cudaFree(d_blkcolidxC);
    cudaFree(d_nnzb_C);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (numblkC+1) * sizeof(int);
    // (*index) += 1;                            

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
unsigned long long int *nnzC_computed, double *compression_rate, double *time_stir, double *gflops_stir,char *filename)
{
//for (int i = 0; i < blkmA; i++)
//    printf("[%i],%i\n", i, blkrowptrA[i+1] - blkrowptrA[i]);

    // struct timeval tv;

    int *d_blkrowptrA;
    int *d_blkcolidxA; 
    int *d_nnzb_A; 
    MAT_VAL_TYPE *d_blkcsr_Val_A;
    unsigned char *d_blkcsr_Col_A;
    unsigned char *d_blkcsr_Ptr_A;

    cudaMalloc((void **)&d_blkrowptrA, (blkmA+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (blkmA+1) * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcolidxA, numblkA * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkA * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_nnzb_A, (numblkA+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (numblkA+1) * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcsr_Val_A, nnzA * sizeof(MAT_VAL_TYPE));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + nnzA * sizeof(MAT_VAL_TYPE);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcsr_Col_A, nnzA * sizeof(unsigned char));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + nnzA * sizeof(unsigned char);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcsr_Ptr_A, numblkA * BLOCK_SIZE  * sizeof(unsigned char));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkA * BLOCK_SIZE  * sizeof(unsigned char);
    // (*index) += 1;

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
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (blknB+1) * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkrowidxB, numblkB * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkB * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkrowptrB, (blkmB+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (blkmB+1) * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcolidxB, numblkB * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkB * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_nnzb_B, (numblkB+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (numblkB+1) * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcsr_Val_B, nnzB * sizeof(MAT_VAL_TYPE));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + nnzB * sizeof(MAT_VAL_TYPE);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcsr_Col_B, nnzB * sizeof(unsigned char));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + nnzB * sizeof(unsigned char);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkcsr_Ptr_B, numblkB * BLOCK_SIZE  * sizeof(unsigned char));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkB * BLOCK_SIZE  * sizeof(unsigned char);
    // (*index) += 1;


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
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkB * BLOCK_SIZE  * sizeof(unsigned short);
    // (*index) += 1;

    cudaMemcpy(d_blkmaskB,     blkmaskB,     numblkB * BLOCK_SIZE * sizeof(unsigned short),              cudaMemcpyHostToDevice);

    unsigned short *d_blkmaskA;
    cudaMalloc((void **)&d_blkmaskA, numblkA * BLOCK_SIZE  * sizeof(unsigned short));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkA * BLOCK_SIZE  * sizeof(unsigned short);
    // (*index) += 1;

    cudaMemcpy(d_blkmaskA,     blkmaskA,     numblkA * BLOCK_SIZE * sizeof(unsigned short),              cudaMemcpyHostToDevice);

int numblkC = 0;
int nnzC = 0;
double stir_spgemm_time = 0;

cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * 4);
for (int i = 0; i < 4; i++) {
    cudaStreamCreate(&(stream[i]));
}

double time_all[REPEAT_NUM];


int *d_blksmem_tny_cnt;
int *d_blksmem_sml_cnt;
int *d_blksmem_lrg_cnt;
int *d_blksmem_dns_cnt;

cudaMalloc((void **)&d_blksmem_tny_cnt, 1 * sizeof(int));
// gettimeofday(&tv, NULL );
// time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
// cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + sizeof(int);
// (*index) += 1;    

cudaMalloc((void **)&d_blksmem_sml_cnt, 1 * sizeof(int));
// gettimeofday(&tv, NULL );
// time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
// cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + sizeof(int);
// (*index) += 1; 

cudaMalloc((void **)&d_blksmem_lrg_cnt, 1 * sizeof(int));
// gettimeofday(&tv, NULL );
// time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
// cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + sizeof(int);
// (*index) += 1; 

cudaMalloc((void **)&d_blksmem_dns_cnt, 1 * sizeof(int));
// gettimeofday(&tv, NULL );
// time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
// cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + sizeof(int);
// (*index) += 1; 

for (int ri = 0; ri < REPEAT_NUM; ri++)
{
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1];
    // (*start_index) = (*index);
    // (*index) += 1;

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
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (blkmA+1) * sizeof(int);
    // (*index) += 1;

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
struct timeval tt1, tt2;
gettimeofday(&tt1, NULL);

sfBIN bin;
    /* Initialize bin */
    init_bin(&bin, blkmA);

    /* Set max bin */
    //set_max_bin(a->d_rpt, a->d_col, b->d_rpt, &bin, M);
    set_max_bin(d_blkrowptrA, d_blkcolidxA, d_blkrowptrB, &bin, blkmA);

    //    cudaMalloc((void **)&d_csrRowPtrC, (blkmA+1) * sizeof(int));
numblkC = 0;
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
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkC * sizeof(int);
    // (*index) += 1;        

    int *d_blkcolidxC;
    cudaMalloc((void **)&d_blkcolidxC, numblkC * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkC * sizeof(int);
    // (*index) += 1;

    unsigned short *d_blkmaskC;
    cudaMalloc((void **)&d_blkmaskC, numblkC *BLOCK_SIZE * sizeof(unsigned short));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkC *BLOCK_SIZE * sizeof(unsigned short);
    // (*index) += 1;
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


    
   //cudaDeviceSynchronize();
    gettimeofday(&tt2, NULL);

    double time_step1 = (tt2.tv_sec - tt1.tv_sec) * 1000.0 + (tt2.tv_usec - tt1.tv_usec) / 1000.0;
 //   printf("step 2 success!\n");

    unsigned char *d_blkcsr_Ptr_C;
    cudaMalloc((void **)&d_blkcsr_Ptr_C, numblkC * BLOCK_SIZE * sizeof(unsigned char));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkC * BLOCK_SIZE * sizeof(unsigned char);
    // (*index) += 1;

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
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + (numblkC+1) * sizeof(int);
    // (*index) += 1;


    //cudaMemcpy(d_blkrowptrC,     blkrowptrC,     (blkmA+1) * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemcpy(d_blkrowidxC,     blkrowidxC,     numblkC * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemcpy(d_blkcolidxC,     blkcolidxC,     numblkC * sizeof(int),              cudaMemcpyHostToDevice);
    //cudaMemset(d_nnzb_C, 0, (numblkC+1) * sizeof(int));

    //struct timeval tm1, tm2;
    //cudaDeviceSynchronize();
    //gettimeofday(&tm1, NULL);

    int *d_blkid_smem_tny;
    int *d_blkid_smem_sml;
    int *d_blkid_smem_lrg;
    int *d_blkid_smem_dns;

    cudaMalloc((void **)&d_blkid_smem_tny, numblkC * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkC * sizeof(int);
    // (*index) += 1;   

    cudaMalloc((void **)&d_blkid_smem_sml, numblkC * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkC * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkid_smem_lrg, numblkC * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkC * sizeof(int);
    // (*index) += 1;

    cudaMalloc((void **)&d_blkid_smem_dns, numblkC * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + numblkC * sizeof(int);
    // (*index) += 1;


    cudaMemset(d_blksmem_tny_cnt, 0, 1 * sizeof(int));
    cudaMemset(d_blksmem_sml_cnt, 0, 1 * sizeof(int));
    cudaMemset(d_blksmem_lrg_cnt, 0, 1 * sizeof(int));
    cudaMemset(d_blksmem_dns_cnt, 0, 1 * sizeof(int));

    //cudaDeviceSynchronize();
    //gettimeofday(&tm2, NULL);
    //double time_mem= (tm2.tv_sec - tm1.tv_sec) * 1000.0 + (tm2.tv_usec - tm1.tv_usec) / 1000.0;
    //printf("cuda mem takes %f ms\n", time_mem);

    struct timeval ttt1, ttt2;

    //cudaDeviceSynchronize();
    gettimeofday(&ttt1, NULL);

    int num_threads = WARP_SIZE * WARP_PER_BLOCK;
/*
    int num_blocks = ceil((double)numblkC/(double)WARP_PER_BLOCK);
    stir_spgemm_step3_cuda_kernel<128,4><<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A, d_blkmaskA,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B, d_blkmaskB,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, d_nnzb_C, d_blkmaskC, 
                              d_blksmem_tny_cnt, d_blksmem_sml_cnt, d_blksmem_lrg_cnt, d_blksmem_dns_cnt, 
                              d_blkid_smem_tny, d_blkid_smem_sml, d_blkid_smem_lrg, d_blkid_smem_dns, numblkC);
*/
    int num_blocks = ceil((double)numblkC/(double)(WARP_PER_BLOCK * TILE_PER_WARP));
    stir_spgemm_step3_cuda_kernel_2level<128, TILE_PER_WARP><<< num_blocks, num_threads >>>
                    (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A, d_blkmaskA,
                    blkmA, blknA, numblkA, nnzA, 
                    d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B, d_blkmaskB,
                    blkmB, blknB, numblkB, nnzB, 
                    d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, d_nnzb_C, d_blkmaskC, 
                    d_blksmem_tny_cnt, d_blksmem_sml_cnt, d_blksmem_lrg_cnt, d_blksmem_dns_cnt, 
                    d_blkid_smem_tny, d_blkid_smem_sml, d_blkid_smem_lrg, d_blkid_smem_dns, numblkC);

exclusive_scan_device_cuda(d_nnzb_C, numblkC+1);
nnzC = 0;
cudaMemcpy(&nnzC,     &d_nnzb_C[numblkC],     sizeof(int), cudaMemcpyDeviceToHost);

    //cudaDeviceSynchronize();
   gettimeofday(&ttt2, NULL);

     double time_step2= (ttt2.tv_sec - ttt1.tv_sec) * 1000.0 + (ttt2.tv_usec - ttt1.tv_usec) / 1000.0;
   
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
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + nnzC * sizeof(unsigned char);
    // (*index) += 1;                            

MAT_VAL_TYPE *d_blkcsr_Val_C;
    cudaMalloc((void **)&d_blkcsr_Val_C, nnzC * sizeof(MAT_VAL_TYPE));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + nnzC * sizeof(MAT_VAL_TYPE);
    // (*index) += 1; 

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

    struct timeval tttt1, tttt2;


int blksmem_tny_cnt = 0;
int blksmem_sml_cnt = 0;
int blksmem_lrg_cnt = 0;
int blksmem_dns_cnt = 0;

cudaMemcpy(&blksmem_tny_cnt,     d_blksmem_tny_cnt,     sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(&blksmem_sml_cnt,     d_blksmem_sml_cnt,     sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(&blksmem_lrg_cnt,     d_blksmem_lrg_cnt,     sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(&blksmem_dns_cnt,     d_blksmem_dns_cnt,     sizeof(int), cudaMemcpyDeviceToHost);

     num_threads = WARP_SIZE * WARP_PER_BLOCK;



// tiny : 1 - 32
    //cudaDeviceSynchronize();
    gettimeofday(&tttt1, NULL);
    if (blksmem_tny_cnt)
    {
        num_blocks = ceil((double)blksmem_tny_cnt/(double)WARP_PER_BLOCK);
        stir_spgemm_step4_cuda_kernel_smem_v2<SMEM_TNY_TH,32><<< num_blocks, num_threads, 0, stream[0] >>>
                                 (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                                  blkmA, blknA, numblkA, nnzA, 
                                  d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                                  blkmB, blknB, numblkB, nnzB, 
                                  d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, 
    d_blkcsr_Col_C,d_blkcsr_Val_C,
    d_nnzb_C, d_blkmaskC, blksmem_tny_cnt, d_blkid_smem_tny);
    }

    //cudaDeviceSynchronize();
    //gettimeofday(&tt2, NULL);
    //double time41 = (tt2.tv_sec - tt1.tv_sec) * 1000.0 + (tt2.tv_usec - tt1.tv_usec) / 1000.0;
    //printf("41 blksmem_tny_cnt = %i, time41 = %.2f ms\n", blksmem_tny_cnt, time41);

// small : 33 - 64
   // cudaDeviceSynchronize();
    //gettimeofday(&tt1, NULL);
    if (blksmem_sml_cnt)
    {
     num_blocks = ceil((double)blksmem_sml_cnt/(double)WARP_PER_BLOCK);
    stir_spgemm_step4_cuda_kernel_smem_v3<SMEM_SML_TH,32,4><<< num_blocks, num_threads, 0, stream[1] >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, 
d_blkcsr_Col_C,d_blkcsr_Val_C,
d_nnzb_C, d_blkmaskC, blksmem_sml_cnt, d_blkid_smem_sml);
                             }
  //  cudaDeviceSynchronize();
  //  gettimeofday(&tt2, NULL);
  //  double time42 = (tt2.tv_sec - tt1.tv_sec) * 1000.0 + (tt2.tv_usec - tt1.tv_usec) / 1000.0;
   // printf("42 blksmem_sml_cnt = %i, time42 = %.2f ms\n", blksmem_sml_cnt, time42);

// large : 65 - 128
   // cudaDeviceSynchronize();
   // gettimeofday(&tt1, NULL);
   if (blksmem_lrg_cnt)
   {
     num_blocks = ceil((double)blksmem_lrg_cnt/(double)WARP_PER_BLOCK);
    stir_spgemm_step4_cuda_kernel_smem_v3<SMEM_LRG_TH,64,4><<< num_blocks, num_threads, 0, stream[2] >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, 
d_blkcsr_Col_C,d_blkcsr_Val_C,
d_nnzb_C, d_blkmaskC, blksmem_lrg_cnt, d_blkid_smem_lrg);
                             }
  //  cudaDeviceSynchronize();
  //  gettimeofday(&tt2, NULL);
  //  double time43 = (tt2.tv_sec - tt1.tv_sec) * 1000.0 + (tt2.tv_usec - tt1.tv_usec) / 1000.0;
    //printf("43 blksmem_lrg_cnt = %i, time43 = %.2f ms\n", blksmem_lrg_cnt, time43);
/*    num_blocks = ceil((double)blkdns_cnt/(double)WARP_PER_BLOCK);

    stir_spgemm_step4_cuda_kernel_dns<<< num_blocks, num_threads >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, 
d_blkcsr_Col_C,d_blkcsr_Val_C,
d_nnzb_C, blkdns_cnt, d_blkid_dns);
*/


// dns : 129 - dns
   // cudaDeviceSynchronize();
   // gettimeofday(&tt1, NULL);
   if (blksmem_dns_cnt)
   {
     num_threads = WARP_SIZE * WARP_PER_BLOCK;
     num_blocks = ceil((double)blksmem_dns_cnt/(double)WARP_PER_BLOCK);
    stir_spgemm_step4_cuda_kernel_dns<128,8><<< num_blocks, num_threads , 0, stream[3] >>>
                             (d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                              blkmA, blknA, numblkA, nnzA, 
                              d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                              blkmB, blknB, numblkB, nnzB, 
                              d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, 
d_blkcsr_Col_C,d_blkcsr_Val_C,
d_nnzb_C, d_blkmaskC, blksmem_dns_cnt, d_blkid_smem_dns);
                             }


    //cudaDeviceSynchronize();
    gettimeofday(&tttt2, NULL);
   double  time_step3 = (tttt2.tv_sec - tttt1.tv_sec) * 1000.0 + (tttt2.tv_usec - tttt1.tv_usec) / 1000.0;
  //  double time44 = (tt2.tv_sec - tt1.tv_sec) * 1000.0 + (tt2.tv_usec - tt1.tv_usec) / 1000.0;
//  printf("ttime_step3 = %.2f ms\n", time_step3);
  // printf("44 blksmem_dns_cnt = %i, time44 = %.2f ms\n", blksmem_dns_cnt, time44);

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
    //*t4 = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_all[ri] = time;
    printf("run %2i takes %f ms \n", ri, time);
    stir_spgemm_time += time;
//stir_spgemm_time = ri >= WARMUP_NUM ? stir_spgemm_time + time : stir_spgemm_time;

  //  printf("time = %.2f ms\n", time);
    

/*if (ri == REPEAT_NUM - 1)
{
    FILE *fout1 = fopen("time.csv", "a");
    if (fout1 == NULL)
        printf("Writing results fails.\n");
    fprintf(fout1, "%f,%f,%f,%f\n",
              time_step1, time_step2, time_step3, time);
    fclose(fout1);
}
*/

cudaFree(d_blkrowptrC);
// gettimeofday(&tv, NULL );
// time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
// cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (blkmA+1) * sizeof(int);
// (*index) += 1;

cudaFree(d_blkrowidxC);
// gettimeofday(&tv, NULL );
// time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
// cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * sizeof(int);
// (*index) += 1;

cudaFree(d_blkcolidxC);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * sizeof(int);
//     (*index) += 1;

cudaFree(d_blkmaskC);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC *BLOCK_SIZE * sizeof(unsigned short);
//     (*index) += 1;

cudaFree(d_nnzb_C);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (numblkC+1) * sizeof(int);
//     (*index) += 1;

cudaFree(d_blkcsr_Ptr_C);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * BLOCK_SIZE * sizeof(unsigned char);
//     (*index) += 1;

cudaFree(d_blkcsr_Col_C);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - nnzC * sizeof(unsigned char);
//     (*index) += 1;

cudaFree(d_blkcsr_Val_C);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - nnzC * sizeof(MAT_VAL_TYPE);
//     (*index) += 1;

cudaFree(d_blkid_smem_tny);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * sizeof(int);
//     (*index) += 1;

cudaFree(d_blkid_smem_sml);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * sizeof(int);
//     (*index) += 1;

cudaFree(d_blkid_smem_lrg);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * sizeof(int);[
//     (*index) += 1;

cudaFree(d_blkid_smem_dns);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * sizeof(int);
//     (*index) += 1;

release_bin(bin);

// printf("ri = %d     RE = %d\n", ri , REPEAT_NUM);
// if (ri != REPEAT_NUM - 1)
// {
//     printf("11111111\n\n\n\n\n\n\n\n\n");
//     cudaFree(d_blkrowptrC);
//     gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (blkmA+1) * sizeof(int);
//     (*index) += 1;

//     cudaFree(d_blkrowidxC);
//     gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * sizeof(int);
//     (*index) += 1;

//     cudaFree(d_blkcolidxC);
//     gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * sizeof(int);
//     (*index) += 1;

//     cudaFree(d_blkmaskC);
//     gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC *BLOCK_SIZE * sizeof(unsigned short);
//     (*index) += 1;

//     cudaFree(d_nnzb_C);
//     gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (numblkC+1) * sizeof(int);
//     (*index) += 1;

//     cudaFree(d_blkcsr_Ptr_C);
//     gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * BLOCK_SIZE * sizeof(unsigned char);
//     (*index) += 1;

//     cudaFree(d_blkcsr_Col_C);
//     gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - nnzC * sizeof(unsigned char);
//     (*index) += 1;

//     cudaFree(d_blkcsr_Val_C);
//     gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - nnzC * sizeof(MAT_VAL_TYPE);
//     (*index) += 1;

//     cudaFree(d_blkid_smem_tny);
//     gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * sizeof(int);
//     (*index) += 1;

//     cudaFree(d_blkid_smem_sml);
//     gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * sizeof(int);
//     (*index) += 1;

//     cudaFree(d_blkid_smem_lrg);
//     gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * sizeof(int);
//     (*index) += 1;

//     cudaFree(d_blkid_smem_dns);
//     gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * sizeof(int);
//     (*index) += 1;

//     cudaFree(d_blksmem_tny_cnt);
//     gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - sizeof(int);
//     (*index) += 1;

//     cudaFree(d_blksmem_sml_cnt);
//     gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - sizeof(int);
//     (*index) += 1;

//     cudaFree(d_blksmem_lrg_cnt);
//     gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - sizeof(int);
//     (*index) += 1;

//     cudaFree(d_blksmem_dns_cnt);
//     gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - sizeof(int);
//     (*index) += 1;

// }
// else if(result_check == 1 && ri == REPEAT_NUM - 1)
// {
//     printf("2222222222\n\n\n\n\n\n\n\n");
//     int *h_blkrowptrC = (int *)malloc((blkmA+1)*sizeof(int));
//     cudaMemcpy(h_blkrowptrC,     d_blkrowptrC,     (blkmA+1) * sizeof(int), cudaMemcpyDeviceToHost);

//     int errcnt = 0;
//     for (int i = 0; i < (blkmA+1); i++)
//         if (h_blkrowptrC[i] != blkrowptrC_golden[i]){ //printf("%i, %i\n", h_blkrowptrC[i], blkrowptrC_golden[i]);
//             errcnt++;}
//     printf("step 1 new, blkrowptrC, errcnt = %i\n", errcnt);

//     int *h_blkcolidxC = (int *)malloc(numblkC*sizeof(int));
//     cudaMemcpy(h_blkcolidxC,     d_blkcolidxC,     numblkC * sizeof(int), cudaMemcpyDeviceToHost);
// /*
//     for(int blki =0;blki < blkmA ;blki ++)
//     {
//         quick_sort_key(h_blkcolidxC + blkrowptrC[blki],blkrowptrC[blki+1] - blkrowptrC[blki]);
//     }

//     cudaMemcpy(d_blkcolidxC,     h_blkcolidxC,     numblkC * sizeof(int), cudaMemcpyHostToDevice);
// */
//      errcnt = 0;
//     for (int i = 0; i < numblkC; i++)
//        if (h_blkcolidxC[i] != blkcolidxC_golden[i])
//             errcnt++;
//     printf("step 2 new, h_blkcolidxC, errcnt = %i\n", errcnt);

//     int *h_nnzb_C = (int *)malloc((numblkC+1)*sizeof(int));
//     cudaMemcpy(h_nnzb_C,     d_nnzb_C,     (numblkC+1) * sizeof(int), cudaMemcpyDeviceToHost);

//      errcnt = 0;
//     for (int i = 0; i < (numblkC+1); i++)
//         if (h_nnzb_C[i] != nnzb_C_golden[i])
//             {//printf("[%i] %i, %i\n", i, h_nnzb_C[i], nnzb_C_golden[i]); 
//           errcnt++;}
//     printf("step 3, h_nnzb_C, errcnt = %i\n", errcnt);

//     MAT_VAL_TYPE *h_blkcsr_Val_C = (MAT_VAL_TYPE *)malloc(nnzC*sizeof(MAT_VAL_TYPE));
//     cudaMemcpy(h_blkcsr_Val_C,     d_blkcsr_Val_C,     nnzC*sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);

//     unsigned char *h_blkcsr_Col_C = (unsigned char *)malloc(nnzC*sizeof(unsigned char));
//     cudaMemcpy(h_blkcsr_Col_C,     d_blkcsr_Col_C,     nnzC*sizeof(unsigned char), cudaMemcpyDeviceToHost);

//     unsigned char *h_blkcsr_Ptr_C = (unsigned char *)malloc(BLOCK_SIZE*numblkC*sizeof(unsigned char));
//     cudaMemcpy(h_blkcsr_Ptr_C,     d_blkcsr_Ptr_C,     BLOCK_SIZE*numblkC*sizeof(unsigned char), cudaMemcpyDeviceToHost);

//      errcnt = 0;
//     for (int i = 0; i < BLOCK_SIZE*numblkC; i++)
//         if (h_blkcsr_Ptr_C[i] != blkcsr_Ptr_C_golden[i])
//             {errcnt++;}
//     printf("step 4, h_blkcsr_Ptr_C, errcnt = %i\n", errcnt);

//     errcnt = 0;
//     for (int i = 0; i < nnzC; i++)
//         if (h_blkcsr_Col_C[i] != blkcsr_Col_C_golden[i])
//             {errcnt++;}
//     printf("step 4, h_blkcsr_Col_C, errcnt = %i\n", errcnt);

//     errcnt = 0;
//     for (int i = 0; i < nnzC; i++)
//         if (h_blkcsr_Val_C[i] != blkcsr_Val_C_golden[i])
//             {//printf("%f, %f\n", h_blkcsr_Val_C[i], blkcsr_Val_C_golden[i]); 
//             errcnt++;}
//     printf("step 4, h_blkcsr_Val_C, errcnt = %i\n", errcnt);

    


// free(h_blkrowptrC);
// free(h_blkcolidxC);
// free(h_nnzb_C);
// free(h_blkcsr_Val_C);
// free(h_blkcsr_Col_C);
// free(h_blkcsr_Ptr_C);

// for (int i = 0; i < 4; i++) {
//     cudaStreamDestroy(stream[i]);
// }
// free(stream);

// cudaFree(d_blkrowptrC);
// gettimeofday(&tv, NULL );
// time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
// cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (blkmA+1) * sizeof(int);
// (*index) += 1;

// cudaFree(d_blkrowidxC);
// gettimeofday(&tv, NULL );
// time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
// cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * sizeof(int);
// (*index) += 1;

// cudaFree(d_blkcolidxC);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * sizeof(int);
//     (*index) += 1;

// cudaFree(d_blkmaskC);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC *BLOCK_SIZE * sizeof(unsigned short);
//     (*index) += 1;

// cudaFree(d_nnzb_C);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (numblkC+1) * sizeof(int);
//     (*index) += 1;

// cudaFree(d_blkcsr_Ptr_C);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * BLOCK_SIZE * sizeof(unsigned char);
//     (*index) += 1;

// cudaFree(d_blkcsr_Col_C);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - nnzC * sizeof(unsigned char);
//     (*index) += 1;

// cudaFree(d_blkcsr_Val_C);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - nnzC * sizeof(MAT_VAL_TYPE);
//     (*index) += 1;

// cudaFree(d_blkid_smem_tny);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * sizeof(int);
//     (*index) += 1;

// cudaFree(d_blkid_smem_sml);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * sizeof(int);
//     (*index) += 1;

// cudaFree(d_blkid_smem_lrg);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * sizeof(int);
//     (*index) += 1;

// cudaFree(d_blkid_smem_dns);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkC * sizeof(int);
//     (*index) += 1;

// cudaFree(d_blksmem_tny_cnt);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - sizeof(int);
//     (*index) += 1;

// cudaFree(d_blksmem_sml_cnt);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - sizeof(int);
//     (*index) += 1;

// cudaFree(d_blksmem_lrg_cnt);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - sizeof(int);
//     (*index) += 1;

// cudaFree(d_blksmem_dns_cnt);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - sizeof(int);
//     (*index) += 1;

// }

    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1];
    // (*end_index) = (*index);
    // (*index) += 1;

}

double time_min = time_all[0];
for (int ri = 1; ri < REPEAT_NUM; ri++)  
    time_min = time_min > time_all[ri] ? time_all[ri] : time_min;

*nnzC_computed = nnzC;
*compression_rate = (double)nnzCub / (double)(*nnzC_computed);
//stir_spgemm_time /= (REPEAT_NUM-WARMUP_NUM); // first WARMUP_NUM runs warm up gpu
//stir_spgemm_time /= REPEAT_NUM; // first WARMUP_NUM runs warm up gpu
stir_spgemm_time = time_min;
*time_stir = stir_spgemm_time;
*gflops_stir = 2.0 * (double)nnzCub / (stir_spgemm_time * 1e6);

printf("CUDA numblkC = %i\n", numblkC);
    printf("CUDA nnzC = %i\n", nnzC);
printf("new stir_spgemm_time = %4.2f ms, gflops = %4.2f\n", stir_spgemm_time, *gflops_stir);

cudaFree(d_blksmem_tny_cnt);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - sizeof(int);
//     (*index) += 1;

cudaFree(d_blksmem_sml_cnt);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - sizeof(int);
//     (*index) += 1;

cudaFree(d_blksmem_lrg_cnt);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - sizeof(int);
//     (*index) += 1;

cudaFree(d_blksmem_dns_cnt);
// gettimeofday(&tv, NULL );
//     time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
//     cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - sizeof(int);
//     (*index) += 1;



    cudaFree(d_blkrowptrA);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (blkmA+1) * sizeof(int);
    // (*index) += 1;

    cudaFree(d_blkcolidxA); 
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkA * sizeof(int);
    // (*index) += 1;

    cudaFree(d_nnzb_A); 
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (numblkA+1) * sizeof(int);
    // (*index) += 1;

    cudaFree(d_blkcsr_Val_A);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - nnzA * sizeof(MAT_VAL_TYPE);
    // (*index) += 1;

    cudaFree(d_blkcsr_Col_A);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - nnzA * sizeof(unsigned char);
    // (*index) += 1;

    cudaFree(d_blkcsr_Ptr_A);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkA * BLOCK_SIZE  * sizeof(unsigned char);
    // (*index) += 1;


    cudaFree(d_blkcolptrB);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (blknB+1) * sizeof(int);
    // (*index) += 1;

    cudaFree(d_blkrowidxB); 
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkB * sizeof(int);
    // (*index) += 1;

    cudaFree(d_blkrowptrB);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (blkmB+1) * sizeof(int);
    // (*index) += 1;

    cudaFree(d_blkcolidxB);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkB * sizeof(int);
    // (*index) += 1;

    cudaFree(d_nnzb_B); 
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - (numblkB+1) * sizeof(int);
    // (*index) += 1;

    cudaFree(d_blkcsr_Val_B);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - nnzB * sizeof(MAT_VAL_TYPE);
    // (*index) += 1;

    cudaFree(d_blkcsr_Col_B);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - nnzB * sizeof(unsigned char);
    // (*index) += 1;

    cudaFree(d_blkcsr_Ptr_B);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkB * BLOCK_SIZE  * sizeof(unsigned char);
    // (*index) += 1;

    cudaFree(d_blkmaskB);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkB * BLOCK_SIZE  * sizeof(unsigned short);
    // (*index) += 1;

    cudaFree(d_blkmaskA);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - numblkA * BLOCK_SIZE  * sizeof(unsigned short);
    // (*index) += 1;

    

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
    //bool check_result = 0;

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

    // Set aside 50% of L2 cache for persisting accesses 
    size_t size = min( int(deviceProp.l2CacheSize * 0.80) , deviceProp.persistingL2CacheMaxSize );
    cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size); 

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

if (rowA == colA && isSymmetricA)
{
printf("matrix AAT does not do symmetric matrix. Exit.\n");
    return 0;
}

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
    //int ptrA_length =0 ;

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
    //int blknnz_C =0;

    int numblkA = rowblock_ptr_A[rbnum_A];
    int numblkB = colblock_ptr_B[cbnum_B];
    int numblkC = 0;

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
    //int ptroffset_B =0;
   

    

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


int *blkcolIdx_C;
int *nnzb_C;
MAT_VAL_TYPE *blkcsr_Val_C;
    unsigned char  *blkcsr_Col_C;
    unsigned char *blkcsr_Ptr_C;
    int nnzC = 0;

#if DEBUG
sfBIN bin;

// --------------------------------------------------------------------------------------------------------
struct timeval tv;
    int cuda_memory_use[1000] = {0};
    double time_node[1000] = {0};
    int index = 1;
    int start_index;
    int end_index;

    // step1 (rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,
    //       colblock_ptr_B,blkrowIdx_B,rbnum_B,cbnum_B,
    //       rowblock_ptr_C, &numblkC);

    //step1_cuda (rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,numblkA,
    //      colblock_ptr_B,blkrowIdx_B,rbnum_B,cbnum_B,numblkB,
    //      rowblock_ptr_C, &numblkC);

    int *d_rowblock_ptr_C;
    cudaMalloc((void **)&d_rowblock_ptr_C, (rbnum_A+1) * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] + (rbnum_A+1) * sizeof(int);
    // (index) += 1;  


    cudaMemset(d_rowblock_ptr_C, 0, (rbnum_A+1) * sizeof(int));


    step1_cuda_new (&bin, rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,numblkA,
          rowblock_ptr_B,blkcolIdx_B,csrrbnum_B,csrcbnum_B,numblkB,
          d_rowblock_ptr_C, rowblock_ptr_C, &numblkC);

    int *rowblock_ptr_C_cuda;
    rowblock_ptr_C_cuda = (int *)malloc((rbnum_A+1) * sizeof (int)) ;
    cudaMemcpy(rowblock_ptr_C_cuda,     d_rowblock_ptr_C,     (rbnum_A+1)*sizeof(int), cudaMemcpyDeviceToHost);

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
    blkcolIdx_C = (int *)malloc(numblkC * sizeof (int)) ;
    printf("rowblock_ptr_C[rbnum_A] = %i\n", rowblock_ptr_C_cuda[rbnum_A]);

    int *d_blkcolIdx_C;
    cudaMalloc((void **)&d_blkcolIdx_C, numblkC * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] + numblkC * sizeof(int);
    // (index) += 1;

    int *d_blkrowIdx_C;
    cudaMalloc((void **)&d_blkrowIdx_C, numblkC * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] + numblkC * sizeof(int);
    // (index) += 1;

    //cudaMemcpy(d_rowblock_ptr_C,     rowblock_ptr_C,     (rbnum_A+1) * sizeof(int),

    //  step2 (rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,
    //        colblock_ptr_B,blkrowIdx_B,rbnum_B,cbnum_B,
    //        rowblock_ptr_C,blkcolIdx_C);

    //step2_cuda (rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,numblkA,
    //      colblock_ptr_B,blkrowIdx_B,rbnum_B,cbnum_B,numblkB,
    //      rowblock_ptr_C,blkcolIdx_C, d_blkrowIdx_C, d_blkcolIdx_C,numblkC);

    step2_cuda_new (&bin, rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,numblkA,
          rowblock_ptr_B,blkcolIdx_B,csrrbnum_B,csrcbnum_B,numblkB,
          rowblock_ptr_C_cuda,blkcolIdx_C, d_blkrowIdx_C, d_blkcolIdx_C,numblkC);

    release_bin(bin);

    printf("step 2 success!\n");
/*for (int i=0;i<rowblock_ptr_C[rbnum_A];i++)
{
    printf("%d     ",blkcolIdx_C[i]);
}
printf("\n");
*/


    nnzb_C = (int *)malloc((rowblock_ptr_C_cuda[rbnum_A] + 1) * sizeof (int)) ;
    memset(nnzb_C,0,(rowblock_ptr_C_cuda[rbnum_A] + 1) * sizeof (int)) ;



    //  step3  (rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,nnzb_A,rowA,
    //          blkcsr_Val_A,blkcsr_Col_A,blkcsr_Ptr_A,
    //         colblock_ptr_B,blkrowIdx_B,rbnum_B,cbnum_B,nnzb_B,
    //         blkcsr_Val_B,blkcsr_Col_B,blkcsr_Ptr_B,
    //          rowblock_ptr_C,blkcolIdx_C,nnzb_C, &nnzC);

//-----------------------------------------------------------------------------------------------------

    unsigned char *d_blkcsr_Ptr_C;
    cudaMalloc((void **)&d_blkcsr_Ptr_C, numblkC * BLOCK_SIZE * sizeof(unsigned char));
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] + numblkC * BLOCK_SIZE * sizeof(unsigned char);
    // (index) += 1;

    //cudaMemset(d_blkcsr_Ptr_C, 0, numblkC * BLOCK_SIZE * sizeof(unsigned char));

    unsigned short *d_blkmaskC;
    cudaMalloc((void **)&d_blkmaskC, numblkC *BLOCK_SIZE * sizeof(unsigned short));
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] + numblkC *BLOCK_SIZE * sizeof(unsigned short);
    // (index) += 1;

    //cudaMemset(d_blkmaskC, 0, numblkC *BLOCK_SIZE * sizeof(unsigned short));

    int *d_blkid_smem_tny;
    int *d_blkid_smem_sml;
    int *d_blkid_smem_lrg;
    int *d_blkid_smem_dns;

    cudaMalloc((void **)&d_blkid_smem_tny, numblkC * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] + numblkC * sizeof(int);
    // (index) += 1;

    cudaMalloc((void **)&d_blkid_smem_sml, numblkC * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] + numblkC * sizeof(int);
    // (index) += 1;

    cudaMalloc((void **)&d_blkid_smem_lrg, numblkC * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] + numblkC * sizeof(int);
    // (index) += 1;

    cudaMalloc((void **)&d_blkid_smem_dns, numblkC * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] + numblkC * sizeof(int);
    // (index) += 1;


    int *d_blksmem_tny_cnt;
    int *d_blksmem_sml_cnt;
    int *d_blksmem_lrg_cnt;
    int *d_blksmem_dns_cnt;

    cudaMalloc((void **)&d_blksmem_tny_cnt, 1 * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] + sizeof(int);
    // (index) += 1;

    cudaMalloc((void **)&d_blksmem_sml_cnt, 1 * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] + sizeof(int);
    // (index) += 1;

    cudaMalloc((void **)&d_blksmem_lrg_cnt, 1 * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] + sizeof(int);
    // (index) += 1;

    cudaMalloc((void **)&d_blksmem_dns_cnt, 1 * sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] + sizeof(int);
    // (index) += 1;

    cudaMemset(d_blksmem_tny_cnt, 0, 1 * sizeof(int));
    cudaMemset(d_blksmem_sml_cnt, 0, 1 * sizeof(int));
    cudaMemset(d_blksmem_lrg_cnt, 0, 1 * sizeof(int));
    cudaMemset(d_blksmem_dns_cnt, 0, 1 * sizeof(int));
    
    
    int *d_nnzb_C;

   
    cudaMalloc((void **)&d_nnzb_C, (numblkC+1) * sizeof(int));
    
    cudaMemset(d_nnzb_C, 0, (numblkC+1) * sizeof(int));


    step3_cuda  (rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,numblkA, nnzA, nnzb_A,rowA,
            blkcsr_Val_A,blk_idx_A,blkcsr_Ptr_A,blkmaskA,
            colblock_ptr_B,blkrowIdx_B,rbnum_B,cbnum_B,numblkB, nnzB, nnzb_B,
            blkcsr_Val_B,blkcsr_Col_B,blkcsr_Ptr_B,blkmaskB,
            rowblock_ptr_C_cuda,blkcolIdx_C, d_blkrowIdx_C, d_blkcolIdx_C, d_blkcsr_Ptr_C, d_blkmaskC, 
            d_nnzb_C, nnzb_C, numblkC, &nnzC, d_blksmem_tny_cnt, d_blksmem_sml_cnt, d_blksmem_lrg_cnt, d_blksmem_dns_cnt, 
                              d_blkid_smem_tny, d_blkid_smem_sml, d_blkid_smem_lrg, d_blkid_smem_dns);


    
    printf("nnzC = %i\n",nnzC) ;

    int *d_nnzb_C_cuda;

    d_nnzb_C_cuda = (int *)malloc((numblkC + 1) * sizeof (int)) ;

    cudaMemcpy(d_nnzb_C_cuda,  d_nnzb_C,  (numblkC + 1)*sizeof(int), cudaMemcpyDeviceToHost);

    int errcnt_3=0;
    for(int i=0;i< numblkC +1;i++)
    {
        if (d_nnzb_C_cuda[i] != nnzb_C[i])
        errcnt_3 ++;
    }

    printf("step3 errcnt = %d\n",errcnt_3);


    printf("step 3 success!\n");


    blkcsr_Val_C=(MAT_VAL_TYPE*)malloc((nnzC)*sizeof(MAT_VAL_TYPE));
    blkcsr_Col_C=(unsigned char*)malloc((nnzC)*sizeof(unsigned char));
    blkcsr_Ptr_C=(unsigned char*)malloc((BLOCK_SIZE * numblkC)*sizeof(unsigned char));
    memset(blkcsr_Ptr_C,0,(BLOCK_SIZE * numblkC)*sizeof(char)) ;

    
    //  step4  (rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,nnzb_A,rowA,
    //      blkcsr_Val_A,blkcsr_Col_A,blkcsr_Ptr_A,
    //      colblock_ptr_B,blkrowIdx_B,rbnum_B,cbnum_B,nnzb_B,
    //      blkcsr_Val_B,blkcsr_Col_B,blkcsr_Ptr_B,
    //      rowblock_ptr_C,blkcolIdx_C,nnzb_C,
    //      blkcsr_Val_C,blkcsr_Col_C,blkcsr_Ptr_C);

   
 
unsigned char *d_blkcsr_Col_C;
    cudaMalloc((void **)&d_blkcsr_Col_C, nnzC * sizeof(unsigned char));
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] + nnzC * sizeof(unsigned char);
    // (index) += 1;

MAT_VAL_TYPE *d_blkcsr_Val_C;
    cudaMalloc((void **)&d_blkcsr_Val_C, nnzC * sizeof(MAT_VAL_TYPE));
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] + nnzC * sizeof(MAT_VAL_TYPE);
    // (index) += 1;

     step4_cuda  (rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,numblkA, nnzA, nnzb_A,rowA,
            blkcsr_Val_A,blk_idx_A,blkcsr_Ptr_A,
            colblock_ptr_B,blkrowIdx_B,rbnum_B,cbnum_B,numblkB, nnzB, nnzb_B,
            blkcsr_Val_B,blkcsr_Col_B,blkcsr_Ptr_B,
            rowblock_ptr_C_cuda,blkcolIdx_C,
d_blkrowIdx_C, d_blkcolIdx_C,
d_blkcsr_Ptr_C, d_blkcsr_Col_C, d_blkcsr_Val_C, d_blkmaskC,
d_nnzb_C_cuda,numblkC,
            blkcsr_Val_C,blkcsr_Col_C,blkcsr_Ptr_C, nnzC, 
d_blksmem_tny_cnt, d_blksmem_sml_cnt, d_blksmem_lrg_cnt, d_blksmem_dns_cnt, 
                              d_blkid_smem_tny, d_blkid_smem_sml, d_blkid_smem_lrg, d_blkid_smem_dns);

    printf("step 4 success!\n");
//return 0;
#endif

double tstep1,tstep2,tstep3,t4;
tstep1=0;tstep2=0;tstep3=0;t4=0;

    printf("\n\nstepall_cuda_new started!\n");
unsigned long long int nnzC_computed;
    double compression_rate = 0;
double time_stir = 0;
    double gflops_stir = 0;
    stepall_cuda_new(rowblock_ptr_A,blkcolIdx_A,rbnum_A,cbnum_A,numblkA, nnzA, nnzb_A,rowA,
            blkcsr_Val_A,blk_idx_A,blkcsr_Ptr_A,blkmaskA,
            colblock_ptr_B,blkrowIdx_B,rowblock_ptr_B,blkcolIdx_B,rbnum_B,cbnum_B,numblkB, nnzB, nnzb_B,
            blkcsr_Val_B,blkcsr_Col_B,blkcsr_Ptr_B,blkmaskB,
            rowblock_ptr_C_cuda,blkcolIdx_C,
            d_nnzb_C_cuda,numblkC,
            blkcsr_Val_C,blkcsr_Col_C,blkcsr_Ptr_C, nnzC, nnzCub, 
&nnzC_computed, &compression_rate, &time_stir, &gflops_stir,filename);
    printf("stepall_cuda_new success!\n");


    cudaFree(d_blkcolIdx_C);
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] - numblkC * sizeof(int);
    // (index) += 1;

    cudaFree(d_blkrowIdx_C);
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] - numblkC * sizeof(int);
    // (index) += 1;

    cudaFree(d_blkcsr_Ptr_C);
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] - numblkC * BLOCK_SIZE * sizeof(unsigned char);
    // (index) += 1;

    cudaFree(d_blkmaskC);
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] - numblkC *BLOCK_SIZE * sizeof(unsigned short);
    // (index) += 1;

    cudaFree(d_blkid_smem_tny);
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] - numblkC * sizeof(int);
    // (index) += 1;

    cudaFree(d_blkid_smem_sml);
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] - numblkC * sizeof(int);
    // (index) += 1;

    cudaFree(d_blkid_smem_lrg);
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] - numblkC * sizeof(int);
    // (index) += 1;

    cudaFree(d_blkid_smem_dns);
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] - numblkC * sizeof(int);
    // (index) += 1;

    cudaFree(d_blksmem_tny_cnt);
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] - sizeof(int);
    // (index) += 1;

    cudaFree(d_blksmem_sml_cnt);
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] - sizeof(int);
    // (index) += 1;

    cudaFree(d_blksmem_lrg_cnt);
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] - sizeof(int);
    // (index) += 1;

    cudaFree(d_blksmem_dns_cnt);
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] - sizeof(int);
    // (index) += 1;

    cudaFree(d_blkcsr_Col_C);
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] - nnzC * sizeof(unsigned char);
    // (index) += 1;

    cudaFree(d_blkcsr_Val_C);
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] - nnzC * sizeof(MAT_VAL_TYPE);
    // // printf("nnzC = %d   mem = %d\n\n", nnzC, sizeof(MAT_VAL_TYPE));
    // (index) += 1;

    cudaFree(d_rowblock_ptr_C);
    // gettimeofday(&tv, NULL );
    // time_node[index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[index] = cuda_memory_use[(index) - 1] - (rbnum_A+1) * sizeof(int);
    // (index) += 1;


    cudaFree(d_nnzb_C);

    // write results to text (scv) file
    FILE *fout = fopen("results_tile.csv", "a");
    if (fout == NULL)
        printf("Writing results fails.\n");
    fprintf(fout, "%s,%i,%i,%i,%lld,%lld,%f,%f,%f\n",
            filename, rowA, colA, nnzA, nnzCub, nnzC_computed, compression_rate, time_stir, gflops_stir);
    fclose(fout);

    // printf("last element = %d           last time = %lf\n", cuda_memory_use[index - 1], time_node[index - 1]);
    // printf("index = %d\n", index);
    // for (int i = 0; i < index; i++)
    // {
    //     printf("index = %d      memuse = %12d    time_node = %12.3lf\n", i, cuda_memory_use[i], time_node[i]);
    // }
    // FILE *fout = fopen("memoryuse_tile.csv", "a");
    // if (fout == NULL)
    //     printf("Writing results fails.\n");
    // fprintf(fout, "%s,%d,%d,%d,", filename, rowA, colA, nnzA);
    // for (int i = start_index; i < end_index; i++)
    // {
    //     fprintf(fout, "%d,%lf,", cuda_memory_use[i], time_node[i]);
    // }
    // fprintf(fout, "%d,%lf\n", cuda_memory_use[end_index], time_node[end_index]);
    // fclose(fout);


   




}

    
