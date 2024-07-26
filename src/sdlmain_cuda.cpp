#include "SDL2/SDL.h"
#include <SDL2/SDL_render.h>
#include <SDL2/SDL_video.h>
#include <format>
#include <stdexcept>

/**
* Shoutouts to this ->
* https://github.com/fsan/cuda_on_sdl
* for the sdl with cuda example
*/

const int SCREEN_HEIGHT = 480;
const int SCREEN_WIDTH = 640;
int main(int argc, const char* argv[]) {



    if (SDL_Init( SDL_INIT_VIDEO ) < 0) {
	throw std::runtime_error(std::format("Error: could not initialize SDL, {}", SDL_GetError()));
    }

    SDL_Window* window = SDL_CreateWindow("Mandelbrot", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    if (!window) {
	throw std::runtime_error(std::format("Error: could not open window, {}", SDL_GetError()));
    }
    SDL_Surface* screen = SDL_CreateRGBSurface(0, SCREEN_WIDTH, SCREEN_HEIGHT, 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000);
    if (!screen) {
	throw std::runtime_error(std::format("Error: could not create screen, {}", SDL_GetError()));
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING | SDL_TEXTUREACCESS_TARGET, SCREEN_WIDTH, SCREEN_HEIGHT);

    if (!texture) {
	throw std::runtime_error(std::format("Error: could not create texture, {}", SDL_GetError()));
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
