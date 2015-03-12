#ifndef __AUTOMATON_VIEWER_H__
#define __AUTOMATON_VIEWER_H__

#include "GLUTWrapper.h"
#include "CellularAutomaton.h"
#include "utils.h"

class AutomatonViewer: public GLUTCallbacks {
public:
	AutomatonViewer(GLUTWrapper *w, CellularAutomaton *a);
	~AutomatonViewer();
	/* GLUTWrapper callbacks */
	void reshape(int width, int height);
	void timer();
	void display();
	void mouse(unsigned buttons, int x, int y);
	void keyPressed(unsigned char ch);
	void menu(const std::string &item);
private:
	void createMenu();
	void updateTitle(long duration);
	/* Performs cellular automata step and updates the title */
	long doIteration();

private:
	unsigned width, height;
	CellularAutomaton *automaton;
	GLUTWrapper *wrapper;
	GLSurface tex;
	bool paused = false;
	float zoomFactor = 1.0f;
	float fillRatio = .2f;
};

#endif /*__AUTOMATON_VIEWER_H__*/
