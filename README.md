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
3. To compile a program using uDNN library 
```bash
nvcc program.cu -lcudnn -o program
```
4. To profile program.cu using nvidia profiler
```bash
nvprof ./program [arg1] [arg2] [arg3]
```
Number of arguments are program specific. Go through the code to know more.
## License
[MIT](https://choosealicense.com/licenses/mit/)
