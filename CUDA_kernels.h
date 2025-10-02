#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>

__global__ void vector_add(const float *A, const float *B, float *C, const int n);

__global__ void convert_rgb_to_grayscale(unsigned char *out, const unsigned char *in, const int width, const int height);

__global__ void blur(unsigned char *out, const unsigned char *in, const int width, const int height);

__global__ void matrix_multiplication(const float *M, const float *N, float *P, const int width);

__global__ void tiled_matrix_multiplication(const float *M, const float *N, float *P, const int width);

#endif