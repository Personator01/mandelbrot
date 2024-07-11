#include <chrono>
#include <cstdio>
#include <format>
#include <print>
#include <string>
#include <ranges>
#include <memory>
#include <vector>
#include <climits>
#include <cmath>
#include "mandelbrot.h"
#include "lib/lodepng/lodepng.h"


std::unique_ptr<std::vector<unsigned char>> get_rgba_img(const std::vector<std::vector<float>>& in);

int main(int argc, char** argv) {
    if (argc < 3 || argc > 7) {
	std::println("Usage: mandelbrot <x size> <y size> [x center (0)] [y center (0)] [scale = 100] [iterations = 10]");
	return -1;
    }
    size_t x_size = std::stoul(argv[1]), y_size = std::stoul(argv[2]);
    if (x_size < 1 || y_size < 1) {
	std::println("x size and y size must be positive integers, given values were x: {}, y: {}", x_size, y_size);
	return -1;
    }
    float x_c = 0, y_c = 0, scale = 100;
    int n_iters = 10;

    if (argc >= 4) x_c = std::stof(argv[3]);
    if (argc >= 5) y_c = std::stof(argv[4]);
    if (argc >= 6) scale = std::stof(argv[5]);
    if (argc >= 7) n_iters = std::stoi(argv[6]);
    if (scale == 0.0) {
	std::println("Scale must be nonzero");
	return -1;
    }

    std::vector v(
	y_size,
	std::vector<float>(x_size, 0.0f)
    );
    Point center{x_c, y_c};

    auto start_t = std::chrono::high_resolution_clock::now();
    calculate_mb(v, center, scale, n_iters);
    auto end_t = std::chrono::high_resolution_clock::now();

    auto dt = end_t - start_t;
    

    std::println("Computation time elapsed: {:%S} seconds", dt);

    std::vector<unsigned char> png_enc;
    std::unique_ptr<std::vector<unsigned char>> rgba = get_rgba_img(v);
    unsigned error = lodepng::encode(png_enc, *rgba, x_size, y_size);
    if (error) {
	std::println("Error encoding image: {}", lodepng_error_text(error));
	return -1;
    }

    lodepng::save_file(png_enc, "out.png");
    //std::fwrite(png_enc.data(), 1, png_enc.size(), stdout); 
}

std::unique_ptr<std::vector<unsigned char>> get_rgba_img(const std::vector<std::vector<float>>& in) {
    auto out = std::make_unique<std::vector<unsigned char>>();
    for (const auto& r : std::ranges::reverse_view(in)) {
	for (const float e : r) {
	    int v = static_cast<unsigned char>(std::round(e * UCHAR_MAX));
	    out->push_back(v); //r
	    out->push_back(v); //g
	    out->push_back(v); //b
	    out->push_back(UCHAR_MAX);//a
	}
    }
    return out;
}
