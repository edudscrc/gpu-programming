#ifndef HIP_KERNELS_H
#define HIP_KERNELS_H

#include <hip/hip_runtime.h>

__global__
void vector_add(const float *A, const float *B, float *C, const int n);

__global__
void convert_rgb_to_grayscale(unsigned char *out, const unsigned char *in, const int width, const int height);

__global__
void blur(unsigned char *out, const unsigned char *in, const int width, const int height);

#endif