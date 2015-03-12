#ifndef __UTILS_H__
#define __UTILS_H__
#include <string>
#include <chrono>
#include <stdint.h>

class GLTexture {
public:
	GLTexture();
	~GLTexture();
	void setSize(unsigned w, unsigned h) { width = w; height = h; }
	void uploadData(unsigned char *);
	inline unsigned getWidth() { return width; }
	inline unsigned getHeight() { return height; }
	inline uint32_t getId() { return texId;}
private:
	uint32_t texId;
	unsigned width, height;
};

class GLSurface: public GLTexture {
public:
	GLSurface():GLTexture(),buf(0) {}
	~GLSurface() { if (buf) delete[] buf; }
	void resize(unsigned,unsigned);
	void putPixel(unsigned, unsigned, uint8_t r, uint8_t g, uint8_t b);
	void update();
private:
	uint8_t *buf;
};

void drawColoredQuad(float x, float y, float w, float h);
void drawTexture(float x, float y, float w, float h, uint32_t textId, float zoomFactor = 1.0);
void setupOrthoCamera(int width, int height);

bool hasOption(int argc, char *argv[], const char *option);

template<typename Resolution, typename T> inline long getDuration(T x) {
	using namespace std::chrono;
	return duration_cast<Resolution>(x).count();
}

template<typename T> class DeleteMe {
public:
	DeleteMe(T *p):ptr(p) {}
	~DeleteMe() { delete ptr; }
private:
	T *ptr;
};

#endif /* __UTILS_H__ */
