#ifndef _CCSB_SPMV_CUDA_
#define _CCSB_SPMV_CUDA_

#include "common.h"
#include "utils.h"
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define WARP_PER_BLOCK 4

__global__ 
void ccsb_spmv_cuda_kernel(const MAT_PTR_TYPE *d_rowblock_ptr,
                           const int          *d_columnid,
                           const int          *d_nnzb_A,
                           const unsigned char         *d_BlockA_Ptr,
                           const unsigned char         *d_BlockA_Col,
                           const MAT_VAL_TYPE *d_BlockA_Val,
                           const MAT_VAL_TYPE *d_x,
                                 MAT_VAL_TYPE *d_y,
                           const int           rbnum)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki = global_id / WARP_SIZE;

    __shared__ MAT_VAL_TYPE s_x[WARP_PER_BLOCK * BLOCK_SIZE];
    
    if (blki < rbnum)
    {
        //printf("gtid = %i, global_warp_id = %i\n", global_id, global_warp_id);

        const int local_warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
        MAT_VAL_TYPE *s_x_warp = &s_x[local_warp_id * BLOCK_SIZE];

        // init y covered by the block row
        char subformat = 0;
        MAT_VAL_TYPE sum = 0;
        
        // for each block in the block row
        for (int blkj = d_rowblock_ptr[blki]; blkj < d_rowblock_ptr[blki+1]; blkj++)
        {
            // if CSR
            if (subformat = 0)
            {
                const int ri  = lane_id / 2;
                const int virtual_lane_id = lane_id % 2;

                // load x needed by this block to SMEM
                const int x_offset = d_columnid[blkj] * BLOCK_SIZE;
                //MAT_VAL_TYPE r_x = 0;
                if (lane_id < BLOCK_SIZE)
                {
                    //s_x[local_warp_id * BLOCK_SIZE + lane_id] = d_x[x_offset + lane_id];
                    s_x_warp[lane_id] = d_x[x_offset + lane_id];
                    //r_x = d_x[x_offset + lane_id];
                }
                //r_x = __shfl_up_sync(0xffffffff, r_x, BLOCK_SIZE);
                
                const int blk_offset = d_nnzb_A[blkj];
                const int start = d_BlockA_Ptr[blkj*BLOCK_SIZE+ri];
                const int stop = ri == BLOCK_SIZE - 1 ? 
                            (d_nnzb_A[blkj+1] - blk_offset) : d_BlockA_Ptr[ri+1+blkj*BLOCK_SIZE];

                for (int rj = start + virtual_lane_id; rj < stop; rj+=2)
                {
                    const char idx = d_BlockA_Col[blk_offset+rj];
                    //const MAT_VAL_TYPE r_x_gathered = __shfl_sync(0xffffffff, r_x, idx);

                    //if (r_x_gathered != s_x_warp[d_BlockA_Col[blk_offset+rj]])
                    //    printf("blkj = %i, lane = %i, idx= %i, r_x_gathered = %4.2f, s_x_smem = %4.2f\n", 
                    //        blkj, lane_id, idx, r_x_gathered, s_x_warp[idx]);
                    
                    //sum += r_x_gathered * d_BlockA_Val[blk_offset+rj];
                    sum += s_x_warp[idx] * d_BlockA_Val[blk_offset+rj];
                    //sum += d_x[x_offset + d_BlockA_Col[blk_offset+rj]] * d_BlockA_Val[blk_offset+rj];
                }
                // fuse sum at virtual_lane_id 0 and 1
                sum += __shfl_down_sync(0xffffffff, sum, 1);

                // move sum usable to the first 16 lanes
                sum = __shfl_down_sync(0xffffffff, sum, lane_id);
            }
            // if ELL 
            // (could use register shuffle for gathering x (when width is multiple of 2), 
            //  since all threads should be active when shuffling (otherwise 
            //  __shfl_sync gives undefined output))
            else if (subformat = 1)
            {
                // load x needed by this block to REGISTER
                const int x_offset = d_columnid[blkj] * BLOCK_SIZE;
                const MAT_VAL_TYPE r_x = lane_id < BLOCK_SIZE ? d_x[x_offset + lane_id] : 0;

                // produce all intermediate products
                const char ell_width = 7;
                const int blk_offset = d_nnzb_A[blkj];
                for (int rj = lane_id; rj < ell_width * BLOCK_SIZE; rj += WARP_SIZE)
                {
                    const char idx = d_BlockA_Col[blk_offset+rj];
                    const MAT_VAL_TYPE r_x_gathered = __shfl_sync(0xffffffff, r_x, idx);
                    sum += r_x_gathered * d_BlockA_Val[blk_offset+rj];
                }
                // fuse sum at virtual_lane_id and it+16
                sum += __shfl_down_sync(0xffffffff, sum, BLOCK_SIZE);
            }
            // if COO
            // the reason of using COO is that the number of nonzeros is very small (no more than 32?), 
            // and the nonzeros should be distributed evenly among the rows (otherwise ELL should be used), 
            // so all nonzeros could be loaded to registers, and atomic add the y to smem
            else if (subformat = 2)
            {
                if (lane_id < BLOCK_SIZE)
                    s_x_warp[lane_id] = 0;

                const int blk_offset = d_nnzb_A[blkj];
                const int stop  = d_nnzb_A[blkj+1];
                if (blk_offset + lane_id < stop)
                {
                    const char idx = d_BlockA_Col[blk_offset+lane_id];
                    const int x_offset = d_columnid[blkj] * BLOCK_SIZE;
                    const MAT_VAL_TYPE r_x = d_x[x_offset + idx & 0x0000000F];
                    atomicAdd(&s_x_warp[(idx >> 4) & 0x0000000F], r_x * d_BlockA_Val[blk_offset+lane_id]);
				}

			    if (lane_id < BLOCK_SIZE)
			        sum += s_x_warp[lane_id];
			}
            // if HYB
            else if (subformat = 3)
            {
                // first do ELL (the above code can be called)

                // then do COO (the above code can be called)

			}

            // if dense row (only work when the #nonzeros in the block is multiple of BLOCK_SIZE)
            // do dot product, and add the result into sum
            else if (subformat = 4)
            {
                if (lane_id < BLOCK_SIZE)
                    s_x_warp[lane_id] = 0;

                // load x needed by this block to REGISTER
                const int x_offset = d_columnid[blkj] * BLOCK_SIZE;
                const MAT_VAL_TYPE r_x = lane_id < BLOCK_SIZE ? d_x[x_offset + lane_id] : 0;
                // copy the x to both half of the 32 registers
                r_x = __shfl_up_sync(0xffffffff, r_x, BLOCK_SIZE);

                // produce all intermediate products
                const char ell_width = 7;
                const int blk_offset = d_nnzb_A[blkj];
                for (int rj = lane_id; rj < ell_width * BLOCK_SIZE; rj += WARP_SIZE)
                {
                    // get products
                    MAT_VAL_TYPE r_product = r_x * d_BlockA_Val[blk_offset+rj];

                    // reduction sum on each half of the warp
                    r_product += __shfl_down_sync(0xffffffff, r_product, 8);
                    r_product += __shfl_down_sync(0xffffffff, r_product, 4);
                    r_product += __shfl_down_sync(0xffffffff, r_product, 2);
                    r_product += __shfl_down_sync(0xffffffff, r_product, 1);

                    // copy the sum to smem
                    if (lane_id % BLOCK_SIZE == 0)
                        s_x_warp[idx_row] = r_product;
                }

			    if (lane_id < BLOCK_SIZE)
			        sum += s_x_warp[lane_id];
			}
            // if dense col (only work when the #nonzeros in the block is multiple of BLOCK_SIZE)
            // do scaling, and add the result into sum
            else if (subformat = 5)
            {
                // load x needed by this block to REGISTER
                const int x_offset = d_columnid[blkj] * BLOCK_SIZE;
                
                // produce all intermediate products
                const char ell_width = 7;
                const int blk_offset = d_nnzb_A[blkj];
                for (int rj = lane_id; rj < ell_width * BLOCK_SIZE; rj += WARP_SIZE)
                {
                    const char idx = lane_id < BLOCK_SIZE ? 0 : 1;
                    const MAT_VAL_TYPE r_x_gathered = d_x[x_offset + idx];
                    sum += r_x_gathered * d_BlockA_Val[blk_offset+rj];
                }
                // fuse sum at virtual_lane_id and it+16
                sum += __shfl_down_sync(0xffffffff, sum, BLOCK_SIZE);
            }
            // if dense (or near dense stored as dense)
            // load x into registers, and do gemm as usual 
            // (the dense block should be stored in column-major)
            else if (subformat = 6)
            {
                // load x needed by this block to REGISTER
                const int x_offset = d_columnid[blkj] * BLOCK_SIZE;
                MAT_VAL_TYPE r_x = lane_id < BLOCK_SIZE ? d_x[x_offset + lane_id] : 0;
                // copy the x to both half of the 32 registers
                r_x = __shfl_up_sync(0xffffffff, r_x, BLOCK_SIZE);

                // produce all intermediate products
                const int blk_offset = d_nnzb_A[blkj];

                //#pragma unroll for () {}
                MAT_VAL_TYPE r_x_gathered;
                int val_offset;
                // round 0
                r_x_gathered = lane_id < BLOCK_SIZE ? __shfl_sync(0xffffffff, r_x,  0) : __shfl_sync(0xffffffff, r_x, 17);
                val_offset = blk_offset+lane_id;
                sum += r_x_gathered * d_BlockA_Val[val_offset];
                // round 1
                r_x_gathered = lane_id < BLOCK_SIZE ? __shfl_sync(0xffffffff, r_x,  2) : __shfl_sync(0xffffffff, r_x, 19);
                val_offset += WARP_SIZE;
                sum += r_x_gathered * d_BlockA_Val[val_offset];
                // round 2
                r_x_gathered = lane_id < BLOCK_SIZE ? __shfl_sync(0xffffffff, r_x,  4) : __shfl_sync(0xffffffff, r_x, 21);
                val_offset += WARP_SIZE;
                sum += r_x_gathered * d_BlockA_Val[val_offset];
                // round 3
                r_x_gathered = lane_id < BLOCK_SIZE ? __shfl_sync(0xffffffff, r_x,  6) : __shfl_sync(0xffffffff, r_x, 23);
                val_offset += WARP_SIZE;
                sum += r_x_gathered * d_BlockA_Val[val_offset];
                // round 4
                r_x_gathered = lane_id < BLOCK_SIZE ? __shfl_sync(0xffffffff, r_x,  8) : __shfl_sync(0xffffffff, r_x, 25);
                val_offset += WARP_SIZE;
                sum += r_x_gathered * d_BlockA_Val[val_offset];
                // round 5
                r_x_gathered = lane_id < BLOCK_SIZE ? __shfl_sync(0xffffffff, r_x, 10) : __shfl_sync(0xffffffff, r_x, 27);
                val_offset += WARP_SIZE;
                sum += r_x_gathered * d_BlockA_Val[val_offset];
                // round 6
                r_x_gathered = lane_id < BLOCK_SIZE ? __shfl_sync(0xffffffff, r_x, 12) : __shfl_sync(0xffffffff, r_x, 29);
                val_offset += WARP_SIZE;
                sum += r_x_gathered * d_BlockA_Val[val_offset];
                // round 7
                r_x_gathered = lane_id < BLOCK_SIZE ? __shfl_sync(0xffffffff, r_x, 14) : __shfl_sync(0xffffffff, r_x, 31);
                val_offset += WARP_SIZE;
                sum += r_x_gathered * d_BlockA_Val[val_offset];

                // fuse sum at virtual_lane_id and it+16
                sum += __shfl_down_sync(0xffffffff, sum, BLOCK_SIZE);
            }

        }
        
        // save sum to d_y
        if (lane_id < BLOCK_SIZE)
            d_y[blki * BLOCK_SIZE + lane_id] = sum;
    }

}

void ccsb_spmv_cuda(const MAT_PTR_TYPE *h_rowblock_ptr,
                    const int          *h_columnid,
                    const int          *h_nnzb_A,
                    const unsigned char         *h_BlockA_Ptr,
                    const unsigned char         *h_BlockA_Col,
                    const MAT_VAL_TYPE *h_BlockA_Val,
                    const MAT_VAL_TYPE *h_x,
                          MAT_VAL_TYPE *h_y_golden,
                    const int           rowA,
                    const int           colA,
                    const int           nnzA,
                    const int           rbnum,
                    const int           nnzbl,
                    const int           ptrA_length)
{
    printf ("rbnum = %i\n", rbnum);
    // copy host mem to device
    MAT_PTR_TYPE *d_rowblock_ptr;
    int *d_columnid;
    int *d_nnzb_A;
    unsigned char *d_BlockA_Ptr;
    unsigned char *d_BlockA_Col;
    MAT_VAL_TYPE *d_BlockA_Val;
    MAT_VAL_TYPE *d_x;
    MAT_VAL_TYPE *d_y;

    cudaMalloc((void **)&d_rowblock_ptr, (rbnum+1) * sizeof(MAT_PTR_TYPE));
    cudaMalloc((void **)&d_columnid,     nnzbl * sizeof(int));
    cudaMalloc((void **)&d_nnzb_A,       (nnzbl+1) * sizeof(int));
    cudaMalloc((void **)&d_BlockA_Ptr,   ptrA_length * sizeof(unsigned char));
    cudaMalloc((void **)&d_BlockA_Col,   nnzA * sizeof(unsigned char));
    cudaMalloc((void **)&d_BlockA_Val,   nnzA * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_x,            colA * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_y,            rowA * sizeof(MAT_VAL_TYPE));

    cudaMemcpy(d_rowblock_ptr, h_rowblock_ptr, (rbnum+1) * sizeof(MAT_PTR_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columnid,     h_columnid,     nnzbl * sizeof(int),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnzb_A,       h_nnzb_A,       (nnzbl+1) * sizeof(int),          cudaMemcpyHostToDevice);
    cudaMemcpy(d_BlockA_Ptr,   h_BlockA_Ptr,   ptrA_length * sizeof(unsigned char),       cudaMemcpyHostToDevice);
    cudaMemcpy(d_BlockA_Col,   h_BlockA_Col,   nnzA * sizeof(unsigned char),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_BlockA_Val,   h_BlockA_Val,   nnzA * sizeof(MAT_VAL_TYPE),      cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,            h_x,            colA * sizeof(MAT_VAL_TYPE),      cudaMemcpyHostToDevice);

    // run spmv
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int num_blocks = ceil ((double)rbnum / (double)(num_threads / WARP_SIZE));

    struct timeval t1, t2;

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        ccsb_spmv_cuda_kernel<<< num_blocks, num_threads >>>
            (d_rowblock_ptr, d_columnid, d_nnzb_A, d_BlockA_Ptr, d_BlockA_Col, d_BlockA_Val,
            d_x, d_y, rbnum);
    }

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    double time_cuda_spmv = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cuda_spmv /= BENCH_REPEAT;
    printf("  CUDA SpMV %4.2f GFlops\n", 2 * (double)nnzA * 1.0e-6 / time_cuda_spmv);

    MAT_VAL_TYPE *h_y = (MAT_VAL_TYPE *)malloc(rowA * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(h_y,            d_y,            rowA * sizeof(MAT_VAL_TYPE),      cudaMemcpyDeviceToHost);

    // check results
    int errcount = 0;
    for (int i = 0; i < rowA; i++)
    {
        if (h_y[i] != h_y_golden[i])
        {
            errcount++;
            //printf("%f    %f,%d\n",h_y[i],h_y_golden[i],i);
        }
    }
    if (errcount == 0)
        printf("  CUDA SpMV PASSED!\n");
    else
        printf("  CUDA SpMV NOT PASSED! errcount = %i\n", errcount);

    cudaFree(d_rowblock_ptr);
    cudaFree(d_columnid);
    cudaFree(d_nnzb_A);
    cudaFree(d_BlockA_Ptr);
    cudaFree(d_BlockA_Col);
    cudaFree(d_BlockA_Val);
    cudaFree(d_x);
    cudaFree(d_y);

    free(h_y);
}

#endif
