#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <fstream>

struct timespec start, hostDataAlloc, cudaDataAlloc, cudaCalc, cudaCopyToHost, finish;
#define FACTOR 1e6

using namespace std;

__global__ void vectorAddition(int* A, int* B, int* R, int N, int NBlocks)
{
    int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
    int stride = blockDim.x*NBlocks;

    while (threadId < N)
    {
        R[threadId] = A[threadId] + B[threadId];
        threadId += stride;
    }
}

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
    int *d_A, *d_B, *d_R;
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
	
	A = (int*) malloc(memorybytes);
	B = (int*) malloc(memorybytes);
	R = (int*) malloc(memorybytes);
	
	for(int i = 0; i < N; i++)
	{
		A[i] = rand() % 100;
		B[i] = rand() % 100;
	}

    clock_gettime(CLOCK_MONOTONIC,&hostDataAlloc);

    cudaMalloc((void**)&d_A, memorybytes);
    cudaMalloc((void**)&d_B, memorybytes);
    cudaMalloc((void**)&d_R, memorybytes);

    cudaMemcpy(d_A, A, memorybytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, memorybytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, R, memorybytes, cudaMemcpyHostToDevice);

    clock_gettime(CLOCK_MONOTONIC,&cudaDataAlloc);

    vectorAddition<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_R, N, NUM_BLOCKS);
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC,&cudaCalc);

    cudaMemcpy(R, d_R, memorybytes, cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC,&cudaCopyToHost);

    
    verify_result(A, B, R, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_R);

    free(A);
    free(B);
    free(R);

    clock_gettime(CLOCK_MONOTONIC,&finish);

    printf("q2 N=%d threads=%d blocks=%d \n",N,NUM_THREADS,NUM_BLOCKS);

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
