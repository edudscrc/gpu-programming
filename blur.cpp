#include "HIP_kernels.h"
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

void inline gpu_assert(hipError_t err) {
    if (err != hipSuccess) {
        printf("%s in %s at line %d\n", hipGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_HIP(val) gpu_assert((val))

int main(void) {
    int width, height, channels;
    const char *filename {"lena.jpg"};

    unsigned char *h_noise_img {stbi_load(filename, &width, &height, &channels, 1)};

    unsigned char *d_noise_img;
    CHECK_HIP(hipMalloc((void**)&d_noise_img, width * height * sizeof(unsigned char)));
    CHECK_HIP(hipMemcpy(d_noise_img, h_noise_img, width * height * sizeof(unsigned char), hipMemcpyHostToDevice));

    stbi_image_free(h_noise_img);

    unsigned char *d_filtered_img;
    CHECK_HIP(hipMalloc((void**)&d_filtered_img, width * height * sizeof(unsigned char)));

    dim3 block_size {32, 32, 1};
    dim3 grid_size {(width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y, 1};
    blur<<<grid_size, block_size>>>(d_filtered_img, d_noise_img, width, height);

    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipFree(d_noise_img));

    unsigned char *h_filtered_img = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    CHECK_HIP(hipMemcpy(h_filtered_img, d_filtered_img, width * height * sizeof(unsigned char), hipMemcpyDeviceToHost));

    CHECK_HIP(hipFree(d_filtered_img));

    const char *output_path {"filtered.png"};
    int status {stbi_write_png(output_path, width, height, channels, h_filtered_img, width)};

    if (status) {
        std::cout << "Succesful!" << std::endl;
    }
    else {
        std::cout << "Error!" << std::endl;
    }

    free(h_filtered_img);

    return 0;
}