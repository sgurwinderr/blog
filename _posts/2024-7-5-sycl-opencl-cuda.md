---
layout: post
title:  "Comparing SYCL, OpenCL, and CUDA: Matrix Multiplication Example"
author: Gurwinder
categories: [ AI ]
image: assets/images/syclcuda.jpg
featured: false
hidden: false
---

Matrix multiplication is a fundamental operation in many scientific and engineering applications, often accelerated using specialized programming models like SYCL, OpenCL, and CUDA. These models provide ways to harness the power of GPUs for parallel computation. Let's explore how matrix multiplication can be implemented in each of these frameworks and compare their approaches.

| Feature               | OpenCL                                              | SYCL                                              | oneDNN                                         | CUDA                                             |
|-----------------------|-----------------------------------------------------|--------------------------------------------------|------------------------------------------------|--------------------------------------------------|
| **Overview**          | Open standard for parallel programming across heterogeneous platforms. | Higher-level programming model for OpenCL, providing single-source C++ development. | Performance library for deep learning applications. | Proprietary parallel computing platform and API developed by NVIDIA. |
| **Programming Language** | C, C++                                            | C++                                               | C++                                            | C, C++                                            |
| **Target Hardware**   | CPUs, GPUs, FPGAs, DSPs, and other accelerators.    | CPUs, GPUs, FPGAs, and other accelerators supported by OpenCL. | CPUs, GPUs (especially Intel hardware).        | NVIDIA GPUs                                       |
| **Portability**       | High, due to support for multiple vendors and platforms. | High, builds on top of OpenCL to enhance portability and ease of use. | High for supported hardware.                   | Low, limited to NVIDIA GPUs.                      |
| **Ease of Use**       | Moderate, requires understanding of parallel computing concepts. | High, provides modern C++ abstractions and easier development. | Moderate, with a focus on performance optimizations for deep learning. | High for developers familiar with CUDA API, but learning curve for new users. |
| **Performance**       | High, but dependent on the quality of the vendor's OpenCL implementation. | High, with performance close to or matching native OpenCL due to low overhead. | High, optimized for deep learning workloads.   | Very high, with optimizations specific to NVIDIA hardware. |
| **Ecosystem**         | Broad, with support from various hardware vendors and extensive tooling. | Growing, supported by the Khronos Group and increasingly adopted in the industry. | Niche, focused on deep learning applications.  | Strong, with extensive libraries and tools provided by NVIDIA. |
| **Development Complexity** | High, due to low-level programming model and need for manual optimizations. | Lower than OpenCL, due to higher-level abstractions and modern C++ features. | Moderate, with APIs designed for deep learning optimizations. | High, due to need to understand GPU architecture and optimizations. |
| **Standardization**   | Yes, maintained by the Khronos Group.               | Yes, maintained by the Khronos Group.            | No, open-source but primarily driven by Intel. | No, proprietary to NVIDIA.                         |


### CUDA

CUDA, developed by NVIDIA, is a widely used parallel computing platform and programming model for NVIDIA GPUs.

![walking]({{ site.baseurl }}/assets/images/cudamapping.png){:style="display:block; margin-left:auto; margin-right:auto"}

In CUDA programming, when executing a kernel on the GPU, understanding threads, blocks, and grids is essential. These concepts help manage and utilize the parallelism offered by the GPU efficiently.

#### Threads
A thread is the smallest unit of execution in CUDA. Each thread executes the same kernel function independently, but with different data. Threads within the same block can cooperate via shared memory and synchronization mechanisms.
#### Blocks
A block is a group of threads that execute the same kernel code simultaneously. Threads within a block can synchronize with each other and share data through shared memory.
Block Dimensions: Each block can have a maximum of 1,024 threads (as of CUDA capability 7.x). Blocks are organized in three dimensions (x, y, and z), allowing up to 1,024 threads per block in each dimension. The total number of threads in a block (blockDim.x * blockDim.y * blockDim.z) should not exceed the maximum block size supported by the GPU architecture.
#### Grids
A grid is a collection of blocks that execute the kernel function. Blocks in a grid can execute independently of each other and are scheduled on streaming multiprocessors (SMs) of the GPU.
Grids are also organized in three dimensions (x, y, and z), allowing up to 65,535 blocks per grid dimension (gridDim.x, gridDim.y, and gridDim.z). The total number of blocks in a grid (gridDim.x * gridDim.y * gridDim.z) depends on the computation and the available resources on the GPU.

#### Relationship and Usage
##### Thread Indexing
Each thread within a grid has a unique index (threadIdx.x, threadIdx.y, threadIdx.z) that identifies its position within its block.
##### Block Indexing
Each block within a grid has a unique index (blockIdx.x, blockIdx.y, blockIdx.z) that identifies its position within the grid.
##### Grid Configuration
When launching a kernel, you specify the dimensions of the grid and the dimensions of each block (dim3 type in CUDA). This configuration determines how the kernel threads are organized and executed on the GPU.


```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixMul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 1024;
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    // Allocate memory on host
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize matrices h_A and h_B

    // Allocate memory on device
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

### OpenCL

OpenCL is an open standard for parallel programming of heterogeneous systems, supporting CPUs, GPUs, and FPGAs.

```cpp
#include <CL/cl.h>
#include <iostream>

#define MAX_SOURCE_SIZE (0x100000)

int main() {
    const int N = 1024;
    float *h_A, *h_B, *h_C;
    cl_mem d_A, d_B, d_C;
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    size_t size = N * N * sizeof(float);
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize matrices h_A and h_B

    // Get platform and device information
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    // Create OpenCL context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create command queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create memory buffers for matrices
    d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &ret);
    d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &ret);
    d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, &ret);

    // Copy matrices from host to device
    ret = clEnqueueWriteBuffer(command_queue, d_A, CL_TRUE, 0, size, h_A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_B, CL_TRUE, 0, size, h_B, 0, NULL, NULL);

    // Create and build the compute program
    const char *source_str = "kernel void matrixMul(global float *A, global float *B, global float *C, int N) { "
                             "   int row = get_global_id(0); "
                             "   int col = get_global_id(1); "
                             "   if (row < N && col < N) { "
                             "       float sum = 0.0f; "
                             "       for (int i = 0; i < N; ++i) { "
                             "           sum += A[row * N + i] * B[i * N + col]; "
                             "       } "
                             "       C[row * N + col] = sum; "
                             "   } "
                             "} ";
    size_t source_size = strlen(source_str);
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the compute kernel
    kernel = clCreateKernel(program, "matrixMul", &ret);

    // Set the arguments for the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_A);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_B);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_C);
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&N);

    // Execute the OpenCL kernel on the list
    size_t global_item_size[2] = { N, N };
    size_t local_item_size[2] = { 16, 16 };
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);

    // Copy result from the memory buffer
    ret = clEnqueueReadBuffer(command_queue, d_C, CL_TRUE, 0, size, h_C, 0, NULL, NULL);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(d_A);
    ret = clReleaseMemObject(d_B);
    ret = clReleaseMemObject(d_C);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

### SYCL

SYCL is a higher-level programming model built on top of OpenCL for C++. It provides a single-source programming model similar to CUDA.


![walking]({{ site.baseurl }}/assets/images/sycl-diagram.jpg){:style="display:block; margin-left:auto; margin-right:auto"}


```cpp
#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

class MatrixMul;

int main() {
    const int N = 1024;
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(N * N * sizeof(float));
    h_B = (float*)malloc(N * N * sizeof(float));
    h_C = (float*)malloc(N * N * sizeof(float));

    // Initialize matrices h_A and h_B

    {
        range<2> r(N, N);
        buffer<float, 2> A(h_A, r);
        buffer<float, 2> B(h_B, r);
        buffer<float, 2> C(h_C, r);

        queue Q;
        Q.submit([&](handler &h) {
            auto a = A.get_access<access::mode::read>(h);
            auto b = B.get_access<access::mode::read>(h);
            auto c = C.get_access<access::mode::write>(h);

            h.parallel_for<class MatrixMul>(r, [=](id<2> idx) {
                int row = idx[0];
                int col = idx[1];
                float sum = 0.0f;
                for (int i = 0; i < N; ++i) {
                    sum += a[{row, i}] * b[{i, col}];
                }
                c[idx] = sum;
            });
        });
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```
##### Sycl Steps
Initialization: Define the size of matrices and create vectors to hold the data.
Setup: Create a SYCL queue and buffers for the input and output matrices.
Kernel Execution: Submit a kernel to the queue that performs the matrix multiplication in parallel using the parallel_for construct.
Synchronization: Wait for the kernel execution to complete.
Completion: The program ends.

### Comparison
#### Programming Model
CUDA: CUDA uses a directive-based approach where annotations are added to C/C++ code to define parallelism explicitly. It offers fine-grained control over GPU resources and optimizations.

OpenCL: OpenCL employs a runtime API with a C-like language for defining parallel computations. It is designed for heterogeneous platforms, allowing code portability across different GPU and CPU architectures.

SYCL: SYCL provides a higher-level abstraction with a single-source programming model in C++. It integrates with standard C++ code, leveraging modern C++ features and ensuring easier portability and maintainability compared to OpenCL.

#### Memory Management
CUDA: Memory management in CUDA involves explicit allocation and deallocation using CUDA API functions (cudaMalloc, cudaMemcpy, cudaFree).

OpenCL: OpenCL uses a command queue for data transfer and kernel execution. Memory buffers are explicitly created and managed using OpenCL API functions (clCreateBuffer, clEnqueueWriteBuffer, clEnqueueReadBuffer).

SYCL: SYCL simplifies memory management with buffers that automatically handle data movement between host and device. It abstracts away much of the low-level memory management, similar to CUDA but integrated with C++ memory handling.

#### Ease of Use and Portability
CUDA: Best suited for NVIDIA GPUs, CUDA offers high performance and deep integration with NVIDIA hardware but lacks portability to other GPU architectures.

OpenCL: OpenCL provides broader platform support across different GPU vendors (NVIDIA, AMD, Intel) and CPUs, making it more portable but potentially less optimized for specific hardware.

SYCL: SYCL inherits portability from OpenCL while providing a more user-friendly programming model with C++ integration. It is ideal for developers familiar with C++ looking for a high-level GPU programming model.

### Conclusion
Choosing between CUDA, OpenCL, and SYCL depends on factors like performance requirements, hardware availability, and developer familiarity. CUDA excels in performance and NVIDIA GPU integration but is less portable. OpenCL offers portability across different platforms but may require more effort for optimization. SYCL combines the portability of OpenCL with a more intuitive C++ programming model, appealing to developers seeking ease of use and maintainability.