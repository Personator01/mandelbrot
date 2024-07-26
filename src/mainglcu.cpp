#include "render.h"
#include <GL/freeglut_std.h>
#include <GL/gl.h>
#include <print>
#include <GL/glu.h>
//#include <GL/freeglut_std.h>
#include "GL/glut.h"

const int WINDOW_WIDTH = 1280;
const int WINDOW_HEIGHT = 720;

Point center{0, 0};
int width{1280};
int height{720};
float scale{100};
int iters{100};

void display();
void dInit();
void keys (unsigned char key, int, int);

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


    texture = init(WINDOW_WIDTH, WINDOW_HEIGHT);

    glutDisplayFunc(display);
    glutKeyboardFunc(keys);
    glutMainLoop();
}

void keys (unsigned char key, int x, int y) {
    switch (key){
    case ' ':
	    printf("spacebar\n");
    break;
    case 'a':
	    printf("a\n");
    break;
    default:
    break;
    }
    glutPostRedisplay();
}
void display(){
    std::println("2mogus");
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    render(center, width, height, scale, iters);
    printf("%d\n", texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    /**
    glBegin(GL_QUADS);
    glColor3f(1.0f, 0.0f, 0.0f);   glVertex2f(-1,   -1);
    glColor3f(0.0f, 1.0f, 0.0f);   glVertex2f(-1,  1);
    glColor3f(0.0f, 0.0f, 1.0f);   glVertex2f(1, 1);
    glColor3f(1.0f, 0.0f, 0.0f);   glVertex2f(1,   -1);
    glEnd();
    */
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);   glVertex2f(-1,   -1);
    glTexCoord2f(0, 1);   glVertex2f(-1,  1);
    glTexCoord2f(1, 0.5);   glVertex2f(1, 1);
    glTexCoord2f(1, 0);   glVertex2f(1,   -1);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);


    glutSwapBuffers();
}


void dInit() {
    /**
    glClearColor(0.0, 0.0, 0.0, 0.0);

    glPointSize(1.0); 
    glMatrixMode(GL_PROJECTION);  
    glLoadIdentity();
    gluOrtho2D(-50,50,-50,50);
    */
}
