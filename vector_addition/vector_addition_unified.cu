// This program computer the sum of two N-element vectors using unified memory

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <fstream>

struct timespec start, hostDataAlloc, cudaDataAlloc, cudaCalc, cudaCopyToHost, finish;
#define FACTOR 1e6

using namespace std;

// CUDA kernel for vector addition
// __global__ means the function is called from the CPU, and runs on the GPU
__global__ void vectorAddition(int* A, int* B, int* R, int N, int NBlocks)
{
    int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
    int stride = blockDim.x * NBlocks;

    while (threadId < N)
    {
        R[threadId] = A[threadId] + B[threadId];
        threadId += stride;
    }
}

// Check vectorAddition result
void verify_result(int* A, int* B, int* R, int N) 
{
    for (int i = 0; i < N; i++) 
    {
        assert(R[i] == A[i] + B[i]);
    }
}

int main(int argc, char* argv[])
{
    clock_gettime(CLOCK_MONOTONIC,&start);

    int *A, *B, *R;
    double time_usec = 0;

    int K = atoi(argv[1]);
    int N = K*FACTOR;

    int NUM_THREADS = atoi(argv[2]);
    int NUM_BLOCKS = 0;

    if (argc == 4)
        NUM_BLOCKS = atoi(argv[3]);
    else
        NUM_BLOCKS = (N + NUM_THREADS - 1)/NUM_THREADS;    
    
    size_t memorybytes = sizeof(int)*N;
	
	// Allocate unified memory
	cudaMallocManaged(&A, memorybytes);
	cudaMallocManaged(&B, memorybytes);
	cudaMallocManaged(&R, memorybytes);
	
	for(int i = 0; i < N; i++)
	{
		A[i] = rand() % 100;
		B[i] = rand() % 100;
	}

    clock_gettime(CLOCK_MONOTONIC,&hostDataAlloc);

    clock_gettime(CLOCK_MONOTONIC,&cudaDataAlloc);

    // Launch the kernel on the GPU
    // Kernel calls are asynchronous (the CPU program continues execution after
    // call, but no necessarily before the kernel finishes)
    vectorAddition<<<NUM_BLOCKS, NUM_THREADS>>>(A, B, R, N, NUM_BLOCKS);

    // Since Kernel calls are asynchronous we wait here. Its necessary for 
    // this particular case because we no longer use cudaMemcpy and hence we 
    // need to manually create a synchronization barrier.
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC,&cudaCalc);

    clock_gettime(CLOCK_MONOTONIC,&cudaCopyToHost);

    // Check result for errors
    verify_result(A, B, R, N);

    // Free memory on device and host
    cudaFree(A);
    cudaFree(B);
    cudaFree(R);

    clock_gettime(CLOCK_MONOTONIC,&finish);

    // Profiling results
    printf("N=%d threads=%d blocks=%d \n",N,NUM_THREADS,NUM_BLOCKS);

    time_usec =(((double)finish.tv_sec *1000000 + (double)finish.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
	printf("Total execution time: %.03lf\n", time_usec/1000);

	time_usec =(((double)hostDataAlloc.tv_sec *1000000 + (double)hostDataAlloc.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
	printf("Total host Data Allocation Time: %.03lf\n", time_usec/1000);
	
	time_usec =(((double)cudaDataAlloc.tv_sec *1000000 + (double)cudaDataAlloc.tv_nsec/1000) - ((double)hostDataAlloc.tv_sec *1000000 + (double)hostDataAlloc.tv_nsec/1000));
	printf("Total Time to copy data to Device: %.03lf\n", time_usec/1000);

	time_usec =(((double)cudaCalc.tv_sec *1000000 + (double)cudaCalc.tv_nsec/1000) - ((double)cudaDataAlloc.tv_sec *1000000 + (double)cudaDataAlloc.tv_nsec/1000));
	printf("Total Time to perform calculations: %.03lf\n", time_usec/1000);

    time_usec =(((double)cudaCopyToHost.tv_sec *1000000 + (double)cudaCopyToHost.tv_nsec/1000) - ((double)cudaCalc.tv_sec *1000000 + (double)cudaCalc.tv_nsec/1000));
	printf("Total Time to copy data to host: %.03lf\n", time_usec/1000);

    return 0;
}
