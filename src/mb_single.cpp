#include <stdexcept>
#include <complex>
#include "mandelbrot.h"
#define proportion_curve(in) in

//possibly openmp?
std::complex<float> screen_space_to_complex(const int x, const int y, const int x_c, const int y_c, const Point<float> center, const float scale) {
    float re = ((x - x_c) / scale) + center.x;
    float im = (y - y_c) / scale + center.y;
    return std::complex{ re, im };
}


float do_mb(const std::complex<float> p, const int n_iters) {
    std::complex<float> c = p;
    std::complex<float> z{0, 0};
    int iters = 0;
    while (iters < n_iters && std::abs(z) < 2) {
	z = std::pow(z, 2) + c;
	iters++;
    }

    return proportion_curve((float) iters / n_iters);
}

void calculate_mb(std::vector<std::vector<float>>& in_v, const Point<float> center, const float scale, const int n_iters) {
    int n_rows = in_v.size();
    if (n_rows < 1) {
	throw std::invalid_argument("vector's row count must be more than 0");
    }
    int n_cols = in_v[0].size();
    if (n_cols < 1) {
	throw std::invalid_argument("vector's column count must be more than 0");
    }
    int x_c = n_cols / 2, y_c = n_cols / 2;
    for (int row = 0; row < n_rows; row++) {
	for (int col = 0; col < n_cols; col++) {
	    in_v[row][col] = do_mb(screen_space_to_complex(col, row, x_c, y_c, center, scale), n_iters);
	}
    }
}

