# CUDA Essentials
This repository is a good starting point to learn about CUDA basics.
Details about CUDA specific syntaxes are added as comments.
Currently I have covered vector addition and 2D convolution operation.

## Content

1. Vector Addition - Baseline, Pinned Memory, Unified Memory
2. Convolution - Baseline, Tiled, cuDNN

## Running Instructions
1. Requires CUDA installed in the machine. We use CUDA v11.7
2. To compile the programs use 'nvcc x.cu -o x'
3. To include cuDNN library add '-lcudnn'
4. To profile using nvidia profiler run it as 'nvprof ./x {args}' 
