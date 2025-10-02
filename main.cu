#include <iostream>
#include <chrono>
#include "CUDA_kernels.h"

void inline gpu_assert(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(val) gpu_assert((val))

int main() {
    int n = 20000;

    int total_size {n * n};
    std::cout << total_size << std::endl;

    float *h_A = (float*)malloc(n * n * sizeof(float));
    float *h_B = (float*)malloc(n * n * sizeof(float));

    for (int i = 0; i < n * n; ++i) {
        h_A[i] = i;
        h_B[i] = 5;
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_B, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_C, n * n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice));

    free(h_A);
    free(h_B);

    dim3 block_size_2d {32, 32};
    dim3 grid_size_2d {
        (n + block_size_2d.x - 1) / block_size_2d.x,
        (n + block_size_2d.y - 1) / block_size_2d.y
    };

    auto t0 {std::chrono::steady_clock::now()};

    matrix_multiplication<<<grid_size_2d, block_size_2d>>>(d_A, d_B, d_C, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    auto t1 {std::chrono::steady_clock::now()};
    auto duration {t1 - t0};
    auto duration_ms {std::chrono::duration_cast<std::chrono::milliseconds>(duration)};
    std::cout << duration_ms.count() / 1000.f << " s\n";

    block_size_2d = {16, 16};
    grid_size_2d = {
        (n + block_size_2d.x - 1) / block_size_2d.x,
        (n + block_size_2d.y - 1) / block_size_2d.y
    };

    t0 = std::chrono::steady_clock::now();

    tiled_matrix_multiplication<<<grid_size_2d, block_size_2d>>>(d_A, d_B, d_C, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    t1 = std::chrono::steady_clock::now();
    duration = t1 - t0;
    duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    std::cout << duration_ms.count() / 1000.f << " s\n";

    return 0;
}