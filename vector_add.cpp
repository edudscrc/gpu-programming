#include "HIP_kernels.h"

void inline gpu_assert(hipError_t err) {
    if (err != hipSuccess) {
        printf("%s in %s at line %d\n", hipGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_HIP(val) gpu_assert((val))

int main(void) {
    int n = 1000;

    float *h_A = (float*)malloc(n * sizeof(float));
    float *h_B = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        h_A[i] = i;
        h_B[i] = 5;
    }

    float *d_A, *d_B, *d_C;
    CHECK_HIP(hipMalloc((void **)&d_A, n * sizeof(float)));
    CHECK_HIP(hipMalloc((void **)&d_B, n * sizeof(float)));
    CHECK_HIP(hipMalloc((void **)&d_C, n * sizeof(float)));

    CHECK_HIP(hipMemcpy(d_A, h_A, n * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B, n * sizeof(float), hipMemcpyHostToDevice));

    free(h_A);
    free(h_B);

    dim3 block_dim {32, 1, 1};
    dim3 grid_dim {(n + block_dim.x - 1) / block_dim.x, 1, 1};

    vector_add<<<grid_dim, block_dim>>>(d_A, d_B, d_C, n);

    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipFree(d_A));
    CHECK_HIP(hipFree(d_B));

    float *h_C = (float*)malloc(n * sizeof(float));

    CHECK_HIP(hipMemcpy(h_C, d_C, n * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(d_C));

    for (int i = 0; i < n; ++i) {
        printf("%f\n", h_C[i]);
    }

    free(h_C);

    return 0;
}