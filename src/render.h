#ifndef RENDER_H
#define RENDER_H
#include <GL/glew.h>
#include <GL/gl.h>
#include "mandelbrot.h"
GLuint init(int height, int width);

template <typename T>
void render(Point<T> center, int width, int height, T scale, int iters);
#endif
