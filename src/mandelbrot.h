#ifndef MANDELBROT_H
#define MANDELBROT_H
#include <vector>
#include <concepts>

struct Point {
    float x;
    float y;
};


/**
* Calculates value of each point in a 2d vector and sets the element to that value.
* center is the point on the complex plane which corresponds to the center (or a pixel away from the center) of the vectors.
* scale is the ratio of pixels in screen space to distance on the complex plane.
* Sets each element of the 2d vector within the range 0.0-1.0 based on how quickly it converges to 0.
*/
void calculate_mb(std::vector<std::vector<float>>& out_v, const Point center, const float scale, int n_iters = 10); 
#endif
