#include "render.h"
#include <GL/freeglut_std.h>
#include <GL/gl.h>
#include <climits>
#include <print>
#include <chrono>
#include <GL/glu.h>
//#include <GL/freeglut_std.h>
#include "GL/glut.h"

const int WINDOW_WIDTH = 1280;
const int WINDOW_HEIGHT = 720;

Point<double> center{0, 0};
int width{WINDOW_WIDTH};
int height{WINDOW_HEIGHT};
double scale{100};
int iters{100};

void display();
void dInit();
void keys (unsigned char key, int, int);
void specialKeys (int key, int, int);

const float movement_factor = 20;

void GLAPIENTRY MessageCallback( GLenum source,
                 GLenum type,
                 GLuint id,
                 GLenum severity,
                 GLsizei length,
                 const GLchar* message,
                 const void* userParam ) {
  fprintf( stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
           ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "" ),
            type, severity, message );
}

GLuint texture;

struct cudaGraphicsResource* positions;
int main(int argc, char** argv) {
    glutInit(&argc, argv);


    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutInitWindowPosition(200, 200);

    glutCreateWindow("Mandelbrot");

    printf("%s\n", glGetString(GL_VERSION));

    printf("%s\n", glewGetErrorString(glewInit()));

    glEnable              ( GL_DEBUG_OUTPUT );
    glDebugMessageCallback( MessageCallback, 0 );

    dInit();

    texture = init(width, height);

    glutDisplayFunc(display);
    glutKeyboardFunc(keys);
    glutSpecialFunc(specialKeys);
    glutMainLoop();
}

void keys (unsigned char key, int x, int y) {
    switch (key){
    case ' ':
	    printf("spacebar\n");
    break;
    case '-':
	scale *= 0.90;
    break;
    case '=':
	scale *= 1.10;
    break;
    case ',':
	if (iters > 1) {
	    if (iters < 25) {
		iters--;
	    } else if (iters < 100) {
		iters -= 5;
	    } else {
		iters -= 10;
	    }
	}
	break;
    case '.':
	if (iters < INT_MAX - 10) {
	    if (iters < 25) {
		iters++;
	    } else if (iters < 100) {
		iters += 5;
	    } else {
		iters += 10;
	    }
	}
	break;
    case 'r':
	glutReshapeWindow(WINDOW_WIDTH, WINDOW_HEIGHT);
    }
    glutPostRedisplay();
}

void specialKeys(int key, int x, int y) {
    switch (key) {
	case GLUT_KEY_LEFT:
	    center.x -= movement_factor / scale;
	break;
	case GLUT_KEY_RIGHT:
	    center.x += movement_factor / scale;
	break;
	case GLUT_KEY_UP:
	    center.y += movement_factor / scale;
	break;
	case GLUT_KEY_DOWN:
	    center.y -= movement_factor / scale;
	break;
    }
    glutPostRedisplay();
}

Point<float> get_float_center() {
    return {static_cast<float>(center.x), static_cast<float>(center.y)};
}

void display(){
    std::println("2mogus");
    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    auto start_t = std::chrono::high_resolution_clock::now();

    if (scale > 5e6) {
	render(center, width, height, scale, iters);
    } else {
	render(get_float_center(), width, height, static_cast<float>(scale), iters);
    }

    auto end_t = std::chrono::high_resolution_clock::now();
    auto dt = end_t - start_t;
    std::println("Computed in {:%S} seconds", dt);

    printf("%d\n", texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);   glVertex2f(-1,   -1);
    glTexCoord2f(0, 1);   glVertex2f(-1,  1);
    glTexCoord2f(1, 1);   glVertex2f(1, 1);
    glTexCoord2f(1, 0);   glVertex2f(1,   -1);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);


    glutSwapBuffers();
}


void dInit() {
    glEnable(GL_TEXTURE_2D);
    glUseProgram(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glDisable(GL_LIGHTING);
    glLoadIdentity();
    glTranslatef(0, 0, 1);
}
