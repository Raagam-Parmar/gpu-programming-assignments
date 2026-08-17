/* 
 * NOTE: According to Nvidia CUDA Progamming Guide, it is not recommended to
 * have global barriers. The following is a quote:
 *
 * > The CUDA programming model enables arbitrarily large grids to run on GPUs
 * > of any size, whether it has only one SM or thousands of SMs. To achieve
 * > this, the CUDA programming model, with some exceptions, requires that
 * > there be no data dependencies between threads in different thread blocks.
 * > That is, a thread should not depend on results from or synchronize with a
 * > thread in a different thread block of the same grid. All the threads
 * > within a thread block run on the same SM at the same time. Different
 * > thread blocks within the grid are scheduled among the available SMs and
 * > may be executed in any order. In short, the CUDA programming model
 * > requires that it be possible to execute thread blocks in any order,
 * > in parallel or in series.
 *
 * If the grid size launched is larger than the number of thread blocks the GPU
 * can schedule concurrently, it will cause a deadlock.
 *
 * For example, setting NUM_BLOCKS to 641 causes a deadlock on Google Colab's
 * GPUs, Tensor T4.
 *
 * Source: https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-blocks-and-grids
 *
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_BLOCKS 4
#define THREADS_PER_BLOCK 256

__device__ void globalBarrier(int *count, int *flag, int numBlocks) {
    // local barrier - ensure all threads in a block reach here
    __syncthreads();

    // local leader - thread 0 coordinates the counter and flag across blocks
    if (threadIdx.x == 0) {
        int ticket = atomicAdd(count, 1);
        if (ticket == numBlocks - 1) {
            // last arriving block resets the count
            *count = 0;

            // ensure memory writes are visible globally
            __threadfence();

            // and set the flag to release all threads
            *flag = 1;
        } else {
            // the other blocks spin until the flag is set
            while (atomicAdd(flag, 0) == 0) { ; }
        }
    }

    __syncthreads();
}

__global__ void testKernel(int *global_count, int *global_flag, int *shared_data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid] = tid;

    globalBarrier(global_count, global_flag, gridDim.x);

    int target_idx = (tid + 128) % (gridDim.x * blockDim.x);
    int read_val = shared_data[target_idx];

    if (tid == 0) {
        printf("Global barrier passed successfully. Thread 0 read target value: %d\n", read_val);
    }
}

int main() {
    int *d_count, *d_flag, *d_data;
    int total_threads = NUM_BLOCKS * THREADS_PER_BLOCK;

    cudaMalloc(&d_count, sizeof(int));
    cudaMalloc(&d_flag, sizeof(int));
    cudaMalloc(&d_data, total_threads * sizeof(int));

    cudaMemset(d_count, 0, sizeof(int));
    cudaMemset(d_flag, 0, sizeof(int));

    testKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_count, d_flag, d_data);
    cudaDeviceSynchronize();

    cudaFree(d_count);
    cudaFree(d_flag);
    cudaFree(d_data);

    return 0;
}
