#ifndef RENDER_H
#define RENDER_H
#include <GL/glew.h>
#include <GL/gl.h>
#include "mandelbrot.h"
GLuint init(int height, int width);

void render(Point center, int width, int height, float scale, int iters);
#endif
