# CUDA Essentials
This repository is a good starting point to learn about CUDA basics.
Details about CUDA specific syntaxes are added as comments.
Currently I have covered vector addition and 2D convolution operation.

## Content

1. Vector Addition - Baseline, Pinned Memory, Unified Memory
2. Convolution - Baseline, Tiled, cuDNN

## Running Instructions
1. Requires CUDA installed in the machine. We use CUDA v11.7
2. To compile the program (program.cu) use 
```bash
nvcc program.cu -o program
```
3. To compile a program using cuDNN library 
```bash
nvcc program.cu -lcudnn -o program
```
4. To profile program.cu using nvidia profiler
```bash
nvprof ./program [arg1] [arg2] [arg3]
```
Number of arguments are program specific. Go through the code to know more.

## Vector Addition Observations:
Tested the code on following cases:
1. 1 block, 1 thread : slight slow down compared to CPU due to lower clock speeds.
2. 1 block, 256 threads : Speedup > 10
3. n* blocks, 256 threads : Speedup > 100 (n is selected such that n * 256 = size of vector)

## 2D Convolution Results
Hardware: Intel Xeon, NVIDIA Tesla P4
| Algorithm             | Runtime (in ms) | Speedup  |
|-----------------------|-----------------|----------|
| Baseline              | 88.626          | -        |
| Tiled                 | 74.520          | 1.1892   |
| CuDNN Library         | 35.914          | 2.4677   |

## License
[MIT](https://choosealicense.com/licenses/mit/)
