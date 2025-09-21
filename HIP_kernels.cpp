#include "HIP_kernels.h"
#include "amd_detail/amd_hip_runtime.h"

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
        int gray_offset {row * width + col};
        
        // RGB image has 3 channels, so it has 3 times more columns than the gray scale image.
        int rgb_offset {gray_offset * 3};

        unsigned char r {in[rgb_offset]};
        unsigned char g {in[rgb_offset + 1]};
        unsigned char b {in[rgb_offset + 2]};

        out[gray_offset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

__global__
void blur(unsigned char *out, const unsigned char *in, const int width, const int height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    // Kernel 3x3
    constexpr int BLUR_SIZE {1};

    if (col < width && row < height) {
        int pixel_value {0};
        int pixels {0};

        for (int blur_row {-BLUR_SIZE}; blur_row < BLUR_SIZE + 1; ++blur_row) {
            for (int blur_col {-BLUR_SIZE}; blur_col < BLUR_SIZE + 1; ++blur_col) {
                int kernel_row {row + blur_row};
                int kernel_col {col + blur_col};

                if (kernel_row >= 0 && kernel_row < height && kernel_col >= 0 && kernel_col < width) {
                    pixel_value += in[kernel_row * width + kernel_col];
                    ++pixels;
                }
            }
        }

        out[row * width + col] = static_cast<unsigned char>(pixel_value / pixels);
    }
}

__global__
void matrix_multiplication(const float *M, const float *N, float *P, const int width) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < width && row < width) {
        float P_value {0.f};

        for (int k {0}; k < width; ++k) {
            P_value += M[row * width + k] * N[k * width + col];
        }

        P[row * width + col] = P_value;
    }
}