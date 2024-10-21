---
layout: post
title:  "Matrix Operations on Intel GPUs: Inside SYCL and PyTorch Kernel Scheduling"
author: Gurwinder
categories: [ Game Development, Unity ]
image: assets/images/intel-arc.webp
featured: false
hidden: false
---

As GPU workloads evolve, understanding how kernels are mapped to hardware resources becomes essential for optimizing performance. Intel GPUs, particularly those using Xe architecture, handle parallel tasks using an efficient scheduling mechanism that organizes workgroups, work-items, and threadgroups. In this article, we’ll delve into how Intel GPUs schedule kernels using SYCL and PyTorch with a focus on SIMD32, workgroups, and threadgroups to perform matrix operations.

We'll walk through a simple matrix addition example on a 2D tensor of size `(256x256)` and explain how the Intel GPU hardware handles this workload.

---

### 1. **Kernel Breakdown: Understanding Workgroups, Work-items, and Threads**

Before jumping into the code, let’s define some key terms related to how the GPU organizes tasks:

- **Work-item**: The smallest unit of computation that a GPU executes. For a matrix operation, each work-item could be responsible for processing one element of the matrix.
  
- **Workgroup**: A collection of work-items that are grouped together and scheduled to execute on the GPU. Workgroups enable the GPU to handle thousands of work-items in parallel.

- **Threadgroup**: In Intel GPUs, threadgroups are collections of threads (each thread can execute one or more work-items) that are mapped to **SIMD (Single Instruction, Multiple Data)** units for parallel execution.

- **SIMD32**: Refers to 32 parallel execution lanes on Intel GPUs. In SIMD32, a group of 32 work-items can be processed simultaneously in a single execution cycle.

#### Workgroup and Threadgroup Configuration:

For the given matrix size of `256x256`, we’ll distribute the workload as follows:

- **Total Work-items**: Each element of the matrix requires a work-item, so the total number of work-items is \(256 \times 256 = 65536\).
- **Workgroups**: The work-items are divided into **workgroups** for better scheduling. In this case, we have 64 workgroups.
- **Local Work-items per Workgroup**: Each workgroup contains 1024 local work-items, organized as \(256 \times 4\).

```cpp
Workload Breakdown:
- Tensor Size: (256, 256)
- Total Work-items: 65536 (256 * 256)
- Workgroups: 64
- Local Work-items per Workgroup: 1024 (256 * 4)
- SIMD32: 32 work-items executed in parallel
```

In Intel’s Xe GPUs, the scheduler must map work-items efficiently across **Execution Units (EUs)** using SIMD units. Let’s explore how this process works:

#### Workgroup Scheduling:
- **Workgroup**: In the SYCL kernel, the 65536 total work-items are divided into **64 workgroups**. Each workgroup has **1024 local work-items**. These are split into smaller units of 256 x 4 for processing.
  
#### Threadgroup and SIMD32:
Intel’s **SIMD32** architecture allows each **threadgroup** to process 32 work-items in parallel. The GPU scheduler maps the workgroups to **Execution Units (EUs)**, which execute the work-items using **SIMD32** units. 

- **SIMD32 Execution**: Each workgroup contains 1024 work-items, which are further broken into 32 work-items executed simultaneously per cycle by SIMD32 lanes. Thus, a total of **32 work-items are executed in parallel** across each SIMD32 unit.
  
```cpp
SIMD32 Breakdown:
- Workgroups: 64
- Local Items per Workgroup: 1024
- SIMD32: 32 work-items executed simultaneously per SIMD32 unit
```

Each SIMD32 unit will process 32 work-items in parallel for every cycle, allowing rapid processing of matrix elements. The **scheduler** will dynamically assign the workgroups to the available **EUs**, ensuring efficient use of GPU resources.

---

### 2. **SYCL Kernel Example: Adding Two Matrices**

Now that we understand the basic building blocks of GPU scheduling, let’s see how SYCL organizes these work-items and workgroups when adding two matrices.

```cpp
#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;

int main() {
    constexpr size_t N = 256;  // Matrix dimensions
    constexpr size_t M = 256;

    // Create SYCL queue
    queue q;

    // Allocate memory for input matrices and result matrix
    float* matrix_a = malloc_shared<float>(N * M, q);
    float* matrix_b = malloc_shared<float>(N * M, q);
    float* result_matrix = malloc_shared<float>(N * M, q);

    // Initialize the matrices
    for (size_t i = 0; i < N * M; ++i) {
        matrix_a[i] = 1.0f;
        matrix_b[i] = 2.0f;
    }

    // Submit the kernel to add the matrices
    q.parallel_for(range<2>(N, M), [=](id<2> idx) {
        size_t i = idx[0] * M + idx[1];  // Flatten the 2D index
        result_matrix[i] = matrix_a[i] + matrix_b[i];
    }).wait();

    // Display the results
    std::cout << "Result matrix [0][0]: " << result_matrix[0] << std::endl;
    std::cout << "Result matrix [255][255]: " << result_matrix[255 * M + 255] << std::endl;

    // Free memory
    free(matrix_a, q);
    free(matrix_b, q);
    free(result_matrix, q);

    return 0;
}
```

In this SYCL code:
- **Work-items**: The `parallel_for` loop is invoked with a 2D range `(256, 256)`, meaning there are 65536 work-items. Each work-item is responsible for adding one element from `matrix_a` and `matrix_b` to store in `result_matrix`.
- **Workgroups**: The 65536 work-items are divided into 64 workgroups, with each workgroup containing 1024 local work-items.

---

### 3. **PyTorch with Intel Extension (IPEX) for GPUs**

Now let’s move on to the PyTorch implementation, where the same matrix operation can be executed on Intel GPUs using **IPEX (Intel Extension for PyTorch)**.

```python
import torch
import intel_extension_for_pytorch as ipex

assert torch.xpu.is_available(), "XPU is not available"

def add_matrices(a, b):
    return a + b

# Initialize matrices on XPU
mat_1 = torch.randn(256, 256, dtype=torch.bfloat16, device="xpu")
mat_2 = torch.randn(256, 256, dtype=torch.bfloat16, device="xpu")

# Compile the kernel with IPEX backend
compiled_add = torch.compile(add_matrices, backend="ipex")
out = compiled_add(mat_1, mat_2)

print(out)
```

#### Key Concepts in PyTorch:
- The matrices are created on the **XPU (Intel GPU)** device.
- Using **torch.compile** with the IPEX backend, the kernel is compiled and optimized for Intel GPUs.
- The PyTorch kernel is scheduled similarly to the SYCL kernel, leveraging SIMD32 units for parallel processing.

---

### Final Words

Intel GPUs utilize a sophisticated kernel scheduling mechanism that distributes workloads across **Execution Units (EUs)**, leveraging **SIMD32** for efficient parallel execution. Both SYCL and PyTorch benefit from Intel’s optimized GPU architecture, which dynamically schedules workgroups and work-items for high-performance matrix operations.

By understanding how **workgroups**, **work-items**, **threadgroups**, and **SIMD32** interact, you can fine-tune your GPU kernels for maximum performance on Intel hardware.