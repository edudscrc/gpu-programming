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
    const char* filename {"peppers.png"};

    unsigned char *h_data {stbi_load(filename, &width, &height, &channels, 3)};

    if (h_data == NULL) {
        std::cerr << "Error: Could not load image " << filename << std::endl;
        return 1;
    }

    std::cout << "Loaded image with size: " << width << "x" << height
              << " and " << channels << " channels" << std::endl;

    unsigned char *d_data;
    CHECK_HIP(hipMalloc((void**)&d_data, width * height * channels * sizeof(unsigned char)));
    CHECK_HIP(hipMemcpy(d_data, h_data, width * height * channels * sizeof(unsigned char), hipMemcpyHostToDevice));

     stbi_image_free(h_data);

    unsigned char *d_grayscale_data;
    CHECK_HIP(hipMalloc((void**)&d_grayscale_data, width * height * sizeof(unsigned char)));

    dim3 block_size {32, 32, 1};
    dim3 grid_size {(width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y, 1};
    convert_rgb_to_grayscale<<<grid_size, block_size>>>(d_grayscale_data, d_data, width, height);
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipFree(d_data));

    unsigned char *h_grayscale_data = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    CHECK_HIP(hipMemcpy(h_grayscale_data, d_grayscale_data, width * height * sizeof(unsigned char), hipMemcpyDeviceToHost));

    CHECK_HIP(hipFree(d_grayscale_data));

    const char* output_png = "gray.png";
    if (!stbi_write_png(output_png, width, height, 1, h_grayscale_data, width)) {
        std::cerr << "Error saving PNG image." << std::endl;
    } else {
        std::cout << "Successfully saved as " << output_png << std::endl;
    }

    free(h_grayscale_data);

    return 0;
}
