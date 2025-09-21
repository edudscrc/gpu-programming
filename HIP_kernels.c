#include "HIP_kernels.h"

__global__
void vector_add(const float *A, const float *B, float *C, const int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__
void convert_rgb_to_grayscale(unsigned char *out, const unsigned char *in, const int width, const int height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < width && row < height) {
        int gray_offset = row * width + col;
        
        // RGB image has 3 channels, so it has 3 times more columns than the gray scale image.
        int rgb_offset = gray_offset * 3;

        unsigned char r = in[rgb_offset];
        unsigned char g = in[rgb_offset + 1];
        unsigned char b = in[rgb_offset + 2];

        out[gray_offset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}
