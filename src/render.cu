#include "render.h"
#include <GL/gl.h>
#include <climits>
#include <cuda_gl_interop.h>
#include <GL/glext.h>
#include <cstdio>
#include <stdexcept>
#include "cuda/std/complex"

#define cu_assert(val) { cu_err((val), __FILE__, __LINE__); }
inline void cu_err(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
	fprintf(stderr, "CUDA: %s %s %d", cudaGetErrorString(code), file, line);
	throw std::runtime_error("Cuda error");
    }
}


unsigned int size;


typedef cuda::std::complex<float> Complex;

__device__ Complex screen_space_to_complex(const int x, const int y, const int x_c, const int y_c, const float x_fc, const float y_fc, const float scale) {
    float re = ((x - x_c) / scale) + x_fc;
    float im = ((y - y_c) / scale) + y_fc;
    return Complex{ re, im };
}

#define proportion_curve(in) in;

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

__global__ void mb_kernel(unsigned char* data, int cx, int cy, float fcx, float fcy, float scale, int n_iters, int n_rows, int n_cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n_rows && c < n_cols) {
	float mb_v = do_mb(screen_space_to_complex(c, r, cx, cy, fcx, fcy, scale), n_iters);
	unsigned char mb_a = static_cast<unsigned char>(mb_v * UCHAR_MAX); 
	data[(r*n_cols + c) * 4 + 0] = mb_a;
	data[(r*n_cols + c) * 4 + 1] = mb_a;
	data[(r*n_cols + c) * 4 + 2] = mb_a;
	data[(r*n_cols + c) * 4 + 3] = 1;
    }
}

void compute_img(unsigned char* dev_ptr, Point center, int width, int height, float scale, int iters) {
    printf("Running cuda calculations\n");
    int x_c = width / 2, y_c = height / 2;

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

    mb_kernel<<<numBlocks, threadsPerBlock>>>(dev_ptr, x_c, y_c, center.x, center.y, scale, iters, height, width);
    cu_assert(cudaPeekAtLastError());
    cu_assert(cudaDeviceSynchronize());

}

GLuint gl_buf;
GLuint gl_texture;
struct cudaGraphicsResource *buf_rs;


void loadTexture(GLuint texture, GLuint buffer, int width, int height) {
    printf("loading texture 1\n");
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void subTexture(GLuint texture, GLuint buffer, int width, int height) {
    printf("loading texture 1\n");
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
}
/**
void loadTestTexture(GLuint texture, int width, int height) {
    printf("loading texture 2\n");
    std::vector<unsigned char> zeroes = std::vector<unsigned char>(size, 0);
    for (int i = 0; i < size / 4; i++) {
	zeroes[4*i] = UCHAR_MAX;
	zeroes[4*i+3] = UCHAR_MAX;
    }
    glBindTexture(GL_TEXTURE_2D, gl_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, zeroes.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void subTestTexture(GLuint texture, int width, int height) {
    printf("sub texture 2\n");
    std::vector<unsigned char> zeroes = std::vector<unsigned char>(size, 0);
    for (int i = 0; i < size / 4; i++) {
	zeroes[4*i] = UCHAR_MAX;
	zeroes[4*i+3] = UCHAR_MAX;
    }
    glBindTexture(GL_TEXTURE_2D, gl_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, zeroes.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
}
*/

GLuint init(int width, int height) {
    size = width * height * 4;

    std::vector<unsigned char> zeroes = std::vector<unsigned char>(size, 0);

    glGenTextures(1, &gl_texture);
    glGenBuffers(1, &gl_buf);

    glBindBuffer(GL_TEXTURE_BUFFER, gl_buf);
    glBufferStorage(GL_TEXTURE_BUFFER, size * sizeof(unsigned char), zeroes.data(), GL_MAP_WRITE_BIT); 
    glBindBuffer(GL_TEXTURE_BUFFER, 0);

    loadTexture(gl_texture, gl_buf, width, height);

    cudaGraphicsGLRegisterBuffer(&buf_rs, gl_buf, cudaGraphicsRegisterFlagsNone);

    return gl_texture;
}

unsigned char* mapped_data;
size_t mapped_size;

void render(Point center, int width, int height, float scale, int iters) {
    cu_assert(cudaGraphicsMapResources(1, &buf_rs));
    cu_assert(cudaGraphicsResourceGetMappedPointer((void**) &mapped_data, &mapped_size, buf_rs));
    if (mapped_size != size * sizeof(unsigned char)) {
	fprintf(stderr, "expected size %lu but got size %lu", size*sizeof(unsigned char), mapped_size);
	throw std::logic_error("error");
    }

    printf("Drawing centered %f, %f, w %d, h %d, scale %f, iters %d\n", center.x, center.y, width, height, scale, iters); 

    compute_img(mapped_data, center, width, height, scale, iters);
    cu_assert(cudaGraphicsUnmapResources(1, &buf_rs));

    subTexture(gl_texture, gl_buf, width, height);
    //subTexture2(gl_texture, width, height);
}


void cleanup() {
}
