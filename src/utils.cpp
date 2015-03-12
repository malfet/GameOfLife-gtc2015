#include "utils.h"
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#endif

#include <assert.h>
#include <strings.h>

/* Primitive RGBA OpenGL texture holder */
GLTexture::GLTexture() {
	glGenTextures(1, &texId);

	glBindTexture(GL_TEXTURE_2D, texId);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
}

GLTexture::~GLTexture() {
	glDeleteTextures(1, &texId);
}

void GLTexture::uploadData(unsigned char *data) {
	glBindTexture(GL_TEXTURE_2D, texId);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	assert (glGetError() == GL_NO_ERROR);
}

/* OpenGL texture with old-school surface as back storage */
void GLSurface::resize(unsigned width, unsigned height) {
	setSize(width,height);
	if (buf) delete[] buf;
	buf = new uint8_t[width*height*4];
}

void GLSurface::update() { uploadData(buf); }
void GLSurface::putPixel(unsigned x, unsigned y, uint8_t r,uint8_t g, uint8_t b) {
	unsigned offs = (y*getWidth()+x)*4;
	buf[offs+0] = r;
	buf[offs+1] = g;
	buf[offs+2] = b;
}

void drawTexture(float x, float y, float w, float h, uint32_t texId, float zf) {
	float tow = w*(zf-1.0)/(2*zf);
	float toh = h*(zf-1.0)/(2*zf);

	glBindTexture(GL_TEXTURE_2D, texId);
	glBegin(GL_QUADS);
	glTexCoord2f(tow, toh);     glVertex3f(x, y, -5);
	glTexCoord2f(w-tow, toh);   glVertex3f(x+w, y, -5);
	glTexCoord2f(w-tow, h-toh); glVertex3f(x+w, y+h, -5);
	glTexCoord2f(tow, h-toh);   glVertex3f(x, y+h, -5);
	glEnd();
}

void setupOrthoCamera(int width, int height) {
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	glOrtho(0, 1, 1, 0, .1, 1000.);
	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();
	glViewport( 0, 0, width, height );
	glEnable(GL_TEXTURE_2D);
}

bool hasOption(int argc, char *argv[], const char *option) {
	for(int i=1; i < argc; ++i)
		if (strcasecmp(argv[i],option) == 0)
			return true;
	return false;
}

