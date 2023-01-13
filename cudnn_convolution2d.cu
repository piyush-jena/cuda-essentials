#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>

#include <cuda.h>
#include <cudnn.h>


using namespace std;

struct timespec start, finish;

double checksum(double *R, int N)
{
    double sum = 0;

    for(int i = 0; i < N ; i++)
    {
        sum += R[i];
    }

    return sum;
}

int main()
{
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    int C = 3, H = 1024, W = 1024;
    int FW = 3, FH = 3, K = 64;
    int P = 1;
    int out_n = 64, out_c = 1, out_h = 1024, out_w = 1024;

    double *tensor, *filter, *R;

    double *d_tensor;
    double *d_filter, *d_R;
    
    size_t ws_size;
    double *ws_data;

    double alpha = 1;
    double beta = 0;

    tensor = (double*) malloc(C*H*W*sizeof(double));
    R = (double*) malloc(K*H*W*sizeof(double));

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

    cudnnTensorDescriptor_t in_desc;
    cudnnTensorDescriptor_t out_desc;
    cudnnFilterDescriptor_t filt_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&in_desc);
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnCreateFilterDescriptor(&filt_desc);
    
    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W);
    cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW);

    cudaMalloc((void**)&d_tensor, 1 * C * H * W * sizeof(double));
    cudaMalloc((void**)&d_filter, K * C * FH * FW * sizeof(double));

    cudaMemcpy(d_tensor, tensor, C*H*W*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, K*C*FW*FH*sizeof(double), cudaMemcpyHostToDevice);

    
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, P, P, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE);
    cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &out_n, &out_c, &out_h, &out_w);

    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, out_n, out_c, out_h, out_w);
    cudaMalloc((void**)&d_R, out_n * out_c * out_h * out_w * sizeof(double));

    const int n_requestedAlgo = 10;
    int n_returnedAlgo;
    cudnnConvolutionFwdAlgoPerf_t fwd_algo_perf[n_requestedAlgo];

    cudnnFindConvolutionForwardAlgorithm (cudnn, in_desc, filt_desc, conv_desc, out_desc, n_requestedAlgo, &n_returnedAlgo, fwd_algo_perf);

    cudnnConvolutionFwdAlgo_t algo;
    algo = fwd_algo_perf[0].algo;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size);
    cudaMalloc(&ws_data, ws_size);

    clock_gettime(CLOCK_MONOTONIC,&start);

    cudnnConvolutionForward(cudnn, &alpha, in_desc, d_tensor, filt_desc, d_filter, conv_desc, algo, ws_data, ws_size, &beta, out_desc, d_R);
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC,&finish);

    cudaMemcpy(R, d_R, out_n*out_c*out_h*out_w*sizeof(double), cudaMemcpyDeviceToHost);

    double time_usec =(((double)finish.tv_sec *1000000 + (double)finish.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
    printf("%.03lf,%.03lf\n", checksum(R, out_n*out_c*out_h*out_w), time_usec/1000);

    // finalizing
    cudaFree(ws_data);
    cudaFree(d_tensor);
    cudaFree(d_filter);
    cudaFree(d_R);

    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    return 0;
}
