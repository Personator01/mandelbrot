#include "mandelbrot.h"
#include <cstdio>
#include <exception>
#include <cuda/std/complex>
#include <stdexcept>
#define proportion_curve(in) in

#define cu_assert(val) { cu_err((val), __FILE__, __LINE__); }
inline void cu_err(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
	fprintf(stderr, "CUDA: %s %s %d", cudaGetErrorString(code), file, line);
	throw new std::exception();
    }
}

typedef cuda::std::complex<float> Complex;

__device__ Complex screen_space_to_complex(const int x, const int y, const int x_c, const int y_c, const float x_fc, const float y_fc, const float scale) {
    float re = ((x - x_c) / scale) + x_fc;
    float im = ((y - y_c) / scale) + y_fc;
    return Complex{ re, im };
}

__device__ float do_mb(const Complex p, const int n_iters) {
    Complex c = p;
    Complex z = Complex{0, 0};
    int iters = 0;
    while (iters < n_iters && cuda::std::abs(z) < 2) {
	z = cuda::std::pow(z, 2.0f) + c;
	iters++;
    }
    return proportion_curve((float) iters / n_iters);
}

__global__ void mb_kernel(float* data, int cx, int cy, float fcx, float fcy, float scale, int n_iters, int n_rows, int n_cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n_rows && c < n_cols) {
	data[r*n_cols + c] = do_mb(screen_space_to_complex(c, r, cx, cy, fcx, fcy, scale), n_iters);
    }
}


void calculate_mb(std::vector<std::vector<float>>& in_v, const Point center, const float scale, const int n_iters) {
    int n_rows = in_v.size();
    if (n_rows < 1) {
	throw std::invalid_argument("vector's row count must be more than 0");
    }
    int n_cols = in_v[0].size();
    if (n_cols < 1) {
	throw std::invalid_argument("vector's column count must be more than 0");
    }
    int x_c = n_cols / 2, y_c = n_rows / 2;

    float* dev_data;
    cu_assert(cudaMalloc(&dev_data, sizeof(float) * n_cols  * n_rows));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(n_cols / threadsPerBlock.x, n_rows / threadsPerBlock.y);

    mb_kernel<<<numBlocks, threadsPerBlock>>>(dev_data, x_c, y_c, center.x, center.y, scale, n_iters, n_rows, n_cols);


    cu_assert(cudaDeviceSynchronize());
    for (int r = 0; r < n_rows; r++) {
	cu_assert(cudaMemcpy(in_v[r].data(), dev_data + r * n_cols, n_cols * sizeof(float), cudaMemcpyDeviceToHost));
    }
    cu_assert(cudaFree(dev_data));
}
