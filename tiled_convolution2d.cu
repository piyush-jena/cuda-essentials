#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

struct timespec start, finish;

__global__ void Conv2d(double* d_tensor, int* d_inputdim, double* d_filter, int* d_filterdim, double* d_R, int* d_outputdim)
{
    int threadId = threadIdx.x;
    int blocksize = blockDim.x;

    extern __shared__ double s_array[];
    
    int K = (blockIdx.x * blockDim.x/(d_outputdim[1]*d_outputdim[2]));
    int slno = blockIdx.x - K*(d_outputdim[1]*d_outputdim[2]/blockDim.x);
    int row = slno / (d_outputdim[2]/blockDim.x);
    int col = slno % (d_outputdim[2]/blockDim.x);

    int x = row;
    int y = col*blockDim.x;

    int finalrowsize = blockDim.x + 2*(d_filterdim[3]/2);

    for (int c = 0; c < d_filterdim[1]; c++)
    {
        for (int i = 0; i < d_filterdim[2]; i++)
        {
            if ((x + i - 1) >= 0 && (y + threadId - 1) >= 0 && (x + i - 1) < d_inputdim[1] && (y + threadId - 1) < d_inputdim[2])
                s_array[c*finalrowsize*d_filterdim[2] + i*finalrowsize + threadId] = d_tensor[c*d_inputdim[1]*d_inputdim[2] + (x + i - 1)*d_inputdim[2] + (y + threadId - 1)];
            else
                s_array[c*finalrowsize*d_filterdim[2] + i*finalrowsize + threadId] = 0;

            if (threadId <= 1)
            {
                if ((x + i - 1) >= 0 && (y + blocksize + threadId - 1) >= 0 && (x + i - 1) < d_inputdim[1] && (y + blocksize + threadId - 1) < d_inputdim[2] && (threadId <= 1))
                    s_array[c*finalrowsize*d_filterdim[2] + i*finalrowsize + threadId + blocksize] = d_tensor[c*d_inputdim[1]*d_inputdim[2] + (x + i - 1)*d_inputdim[2] + (y + blocksize + threadId - 1)];
                else
                    s_array[c*finalrowsize*d_filterdim[2] + i*finalrowsize + threadId + blocksize] = 0;
            }
        }
    }

    __syncthreads();

    double temp = 0;

    for(int i = 0; i < d_filterdim[1]; i++) // channel
    {
        for(int j = 0; j < d_filterdim[2]; j++)
        {
            for(int k = 0; k < d_filterdim[3]; k++)
            {
                temp += d_filter[K*d_filterdim[1]*d_filterdim[2]*d_filterdim[3] + i*d_filterdim[2]*d_filterdim[3] + (d_filterdim[2]-1-j)*d_filterdim[3] + (d_filterdim[3]-1-k)] * s_array[i*finalrowsize*d_filterdim[2] + j*finalrowsize + (k + threadId)];
            }
        }
    }

    d_R[K*d_outputdim[1]*d_outputdim[2] + x*d_outputdim[2] + y + threadId] = temp;
}

double checksum(double *R, int N)
{
    double sum = 0;

    for(int i = 0; i < N ; i++)
    {
        sum += R[i];
    }

    return sum;
}

int main(int argc, char* argv[])
{
    int C = 3, H = 1024, W = 1024;
    int FW = 3, FH = 3, K = 64;
    int P = 1;

    double *tensor, *filter, *R;
    int inputdim[3] = {C,H,W};
    int filterdim[4] = {K,C,FH,FW};
    int outputdim[3] = {K,(H-FH+2*P+1),(W-FW+2*P+1)};

    int NUM_THREADS = 256;
    int NUM_BLOCKS = ((H-FH+2*P+1)*(W-FW+2*P+1)*K/256);
    size_t SHMEM = (NUM_THREADS + 2*(filterdim[3]/2))*filterdim[1]*filterdim[2]*sizeof(double);

    double *d_tensor, *d_filter, *d_R;
    int *d_inputdim;
    int *d_filterdim;
    int *d_outputdim;

    tensor = (double*) malloc(C*H*W*sizeof(double));
    for(int i = 0 ; i < C ; i++)
    {
        for(int j = 0; j < H; j++)
        {
            for(int k = 0; k < W ; k++)
            {
                tensor[i*H*W + j*W + k] = i*(j+k); // ask if i ranges from 0 to N-1 or 1 to N
            }
        }
    }

    filter = (double*) malloc(K*C*FW*FH*sizeof(double));
    for(int i = 0 ; i < K ; i++)
    {
        for(int j = 0; j < C; j++)
        {
            for(int k = 0; k < FH; k++)
            {
                for(int l = 0; l < FW; l++)
                {
                    filter[i*C*FH*FW + j*FH*FW + k*FW + l] = (i+j)*(k+l); // ask if i ranges from 0 to N-1 or 1 to N
                }
            }
        }
    }

    R = (double*) malloc((H-FH+2*P+1)*(W-FW+2*P+1)*K*sizeof(double));
    for(int i = 0 ; i < (H-FH+2*P+1)*(W-FW+2*P+1)*K ; i++)
        R[i] = 0;

    cudaMalloc((void**)&d_tensor, C*H*W*sizeof(double));
    cudaMalloc((void**)&d_inputdim, 3*sizeof(double));
    cudaMalloc((void**)&d_filter, K*C*FH*FW*sizeof(double));
    cudaMalloc((void**)&d_filterdim, 4*sizeof(double));
    cudaMalloc((void**)&d_R, (H-FH+2*P+1)*(W-FW+2*P+1)*K*sizeof(double));
    cudaMalloc((void**)&d_outputdim, 3*sizeof(double));

    cudaMemcpy(d_tensor, tensor, C*H*W*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputdim, inputdim, 3*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, K*C*FW*FH*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filterdim, filterdim, 4*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, R, (H-FH+2*P+1)*(W-FW+2*P+1)*K*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputdim, outputdim, 3*sizeof(double), cudaMemcpyHostToDevice);
    
    clock_gettime(CLOCK_MONOTONIC,&start);

    Conv2d<<<NUM_BLOCKS, NUM_THREADS, SHMEM>>>(d_tensor, d_inputdim, d_filter, d_filterdim, d_R, d_outputdim);
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC,&finish);

    cudaMemcpy(R, d_R, (H-FH+2*P+1)*(W-FW+2*P+1)*K*sizeof(double), cudaMemcpyDeviceToHost); 

    double time_usec =(((double)finish.tv_sec *1000000 + (double)finish.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
    printf("%.03lf,%.03lf\n", checksum(R, (H-FH+2*P+1)*(W-FW+2*P+1)*K), time_usec/1000);

    cudaFree(d_tensor);
    cudaFree(d_inputdim);
    cudaFree(d_filter);
    cudaFree(d_filterdim);
    cudaFree(d_R);
    cudaFree(d_outputdim);
    
    return 0;
}
