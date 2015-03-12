#include "AutomatonViewer.h"
#include "utils.h"
#include <sstream>

AutomatonViewer::AutomatonViewer(GLUTWrapper *w, CellularAutomaton *a): automaton(a),  wrapper(w) {
	createMenu();
	wrapper->setCallbacks (this);
}

AutomatonViewer::~AutomatonViewer() {
	delete automaton;
}

void AutomatonViewer::reshape(int width, int height) {
	setupOrthoCamera (width, height);

	automaton->resize(width, height);
	tex.resize( width, height );
	this->width = width;
	this->height = height;
	automaton->randomize(fillRatio);

	wrapper->startTimer(100);

}

void AutomatonViewer::timer() {
	auto duration = doIteration();
	if (!paused)
		wrapper->startTimer(std::max(duration,(decltype(duration))100));
	wrapper->postRedraw();
}

void AutomatonViewer::display() {
	for (unsigned y = 0; y <height; ++y)
		for (unsigned x = 0; x<width;++x)
			tex.putPixel(x, y, 0, automaton->isCellAlive(x, y) ? 255: 0, 0);
	tex.update();
	drawTexture(0,0,1,1, tex.getId(), zoomFactor);
}

void AutomatonViewer::mouse(unsigned buttons, int x, int y) {
	if (buttons == 0) return;
	float startx = width*(zoomFactor-1)/(2*zoomFactor);
	float starty = height*(zoomFactor-1)/(2*zoomFactor);
	automaton->setCellState(startx+x/zoomFactor, starty+y/zoomFactor, true);
	wrapper->postRedraw();
}

void AutomatonViewer::keyPressed(unsigned char ch) {
	/* Exit on escape */
	if (ch == 27) wrapper->exit(0);
	/* R to start with a new random pattern */
	if (tolower(ch) == 'r') {
		automaton->randomize(fillRatio);
	}

	/* Space to pause */
	if (ch == 32) {
		paused = !paused;
		if (!paused)
			wrapper->startTimer(100);
	}

	if (ch == '+' || ch == '=')
		zoomFactor += .5;
	if ((ch == '-' || ch == '_') && zoomFactor > 1.0)
		zoomFactor -= .5;

	if (paused) {
		updateTitle(0);
		wrapper->postRedraw();
	}
}

void AutomatonViewer::menu(const std::string &item) {
	if (item == "Quit") wrapper->exit(0);
	if (item == "Pause") {
		paused = !paused;
		if (!paused)
			wrapper->startTimer(100);
	}
	if (item == "Zoom in") zoomFactor += .5;
	if (item == "Zoom out" && zoomFactor >1.0) zoomFactor -= .5;
	if (item == "Life (B3/S23)") automaton->setRules("B3/S23");
	if (item == "Inkspot (B3/S012345678)") automaton->setRules("B3/S012345678");
	if (item == "Diamoeba (B35678/S5678)") automaton->setRules("B35678/S5678");
	if (item == "Day & Night (B3678/S34678)") automaton->setRules("B3678/S34678");
	if (item == "Seeds (B2/S)") automaton->setRules("B2/S");

	if (paused) {
		updateTitle(0);
		wrapper->postRedraw();
	}
}

void AutomatonViewer::createMenu() {
	GLUTMenuEntry mainMenu("");
	GLUTMenuEntry rules("Rules");
	rules.addEntry("Life (B3/S23)");
	rules.addEntry("Inkspot (B3/S012345678)");
	rules.addEntry("Diamoeba (B35678/S5678)");
	rules.addEntry("Day & Night (B3678/S34678)");
	rules.addEntry("Seeds (B2/S)");
	mainMenu.addEntry(rules);
	mainMenu.addEntry("Pause");
	mainMenu.addEntry("Zoom in");
	mainMenu.addEntry("Zoom out");
	mainMenu.addEntry("Quit");
	wrapper->createMenu(mainMenu);
}

void AutomatonViewer::updateTitle(long duration) {
	std::ostringstream sstream;
	sstream<<"Game of Life (rules "<<automaton->getRules()<<" ): ";

	if (paused)
		sstream<<" PAUSED ";
	else if (duration > 1000000)
		sstream<<"step took "<<(duration/1000000)<<" msec";
	else if (duration > 1000)
		sstream<<"step took "<<(duration/1000)<<" microsec";
	else
		sstream<<"step took "<<duration<<" nsec";
	sstream<<" zoomFactor="<<zoomFactor;
	wrapper->setTitle(sstream.str());
}

/* Performs cellular automata step and updates the title */
long AutomatonViewer::doIteration() {
	using namespace std::chrono;
	auto start = steady_clock::now();
	automaton->step();
	auto stop = steady_clock::now();
	auto duration = getDuration<nanoseconds>(stop-start);
	updateTitle(duration);
	return duration/1000000;
}

