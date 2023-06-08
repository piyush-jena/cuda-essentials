// This program implements a 2D convolution using CUDA

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

struct timespec start, finish;

//  2-D convolution kernel
//  Arguments:
//      d_tensor   = padded array
//      d_inputdim = [C,H,W]
//      d_filter   = convolution mask/filter
//      d_filterdim= [K,C,FH,FW]
//      d_R        = result array
//      d_outputdim= [K,(H-FH+2*P+1),(W-FW+2*P+1)]
__global__ void Conv2d(double* d_tensor, int* d_inputdim, double* d_filter, int* d_filterdim, double* d_R, int* d_outputdim)
{
    int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;

    int K = (threadId/(d_outputdim[1]*d_outputdim[2]));

    double temp = 0;

    for(int i = 0; i < d_filterdim[1]; i++)
    {
        for(int j = 0; j < d_filterdim[2]; j++)
        {
            for(int k = 0; k < d_filterdim[3]; k++)
            {
                int rx = (threadId - K*d_outputdim[1]*d_outputdim[2])/d_outputdim[2];
                int ry = (threadId - K*d_outputdim[1]*d_outputdim[2])%d_outputdim[2];
                int x = (j-1 + rx); // to be determined from rx,ry and j,k
                int y = (k-1 + ry); // to be determined from rx,ry and j,k

                if ((x >= 0) & (y >= 0) & (x < d_inputdim[1]) & (y < d_inputdim[2]))
                {
                    temp += d_filter[K*d_filterdim[1]*d_filterdim[2]*d_filterdim[3] + i*d_filterdim[2]*d_filterdim[3] + (d_filterdim[2]-1-j)*d_filterdim[3] + (d_filterdim[3]-1-k)]
                    *d_tensor[i*d_inputdim[1]*d_inputdim[2] + x*d_inputdim[2] + y];
                }
            }
        }
    }

    d_R[threadId] = temp;
}

// Prints the checksum. We compare the checksum with the output of cuDNN.
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

    dim3 NUM_THREADS(256);
    dim3 NUM_BLOCKS((H-FH+2*P+1)*(W-FW+2*P+1)*K/256);

    double *d_tensor, *d_filter, *d_R;
    int *d_inputdim;
    int *d_filterdim;
    int *d_outputdim;

    // Allocate the input tensor
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

    // Allocate the filter tensor
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

    // Allocate space for the result
    R = (double*) malloc((H-FH+2*P+1)*(W-FW+2*P+1)*K*sizeof(double));
    for(int i = 0 ; i < (H-FH+2*P+1)*(W-FW+2*P+1)*K ; i++)
        R[i] = 0;

    // Allocate space on the device
    cudaMalloc((void**)&d_tensor, C*H*W*sizeof(double));
    cudaMalloc((void**)&d_inputdim, 3*sizeof(double));
    cudaMalloc((void**)&d_filter, K*C*FH*FW*sizeof(double));
    cudaMalloc((void**)&d_filterdim, 4*sizeof(double));
    cudaMalloc((void**)&d_R, (H-FH+2*P+1)*(W-FW+2*P+1)*K*sizeof(double));
    cudaMalloc((void**)&d_outputdim, 3*sizeof(double));

    // Copy the data to the device
    cudaMemcpy(d_tensor, tensor, C*H*W*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputdim, inputdim, 3*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, K*C*FW*FH*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filterdim, filterdim, 4*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, R, (H-FH+2*P+1)*(W-FW+2*P+1)*K*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputdim, outputdim, 3*sizeof(double), cudaMemcpyHostToDevice);
    
    clock_gettime(CLOCK_MONOTONIC,&start);

    // Call the kernel
    Conv2d<<<NUM_BLOCKS, NUM_THREADS>>>(d_tensor, d_inputdim, d_filter, d_filterdim, d_R, d_outputdim);

    // Since Kernel calls are asynchronous we wait here. Its necessary for 
    // this particular case because we no longer use cudaMemcpy and hence we 
    // need to manually create a synchronization barrier.
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC,&finish);

    // Copy back the result
    cudaMemcpy(R, d_R, (H-FH+2*P+1)*(W-FW+2*P+1)*K*sizeof(double), cudaMemcpyDeviceToHost); 

    double time_usec =(((double)finish.tv_sec *1000000 + (double)finish.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
    printf("%.03lf,%.03lf\n", checksum(R, (H-FH+2*P+1)*(W-FW+2*P+1)*K), time_usec/1000);

    // Free allocated memory on the device
    cudaFree(d_tensor);
    cudaFree(d_inputdim);
    cudaFree(d_filter);
    cudaFree(d_filterdim);
    cudaFree(d_R);
    cudaFree(d_outputdim);
    
    return 0;
}
