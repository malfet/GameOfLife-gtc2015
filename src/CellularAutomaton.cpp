#include "CellularAutomaton.h"
#include <assert.h>
#include <random>
#include <sstream>

/* Implementation of [get|set]Rules() methods of CellularAutomaton class */
void CellularAutomaton::setRules(const std::string &rules) {
	int len = rules.length();
	int cnt = 0;
	birthRules = survivalRules = 0;

	assert (len > 0 && rules[cnt++] == 'B');
	while (cnt < len && isdigit(rules[cnt])) birthRules |= 1<<(rules[cnt++]-'0');
	assert (cnt < len && rules[cnt++] == '/');
	assert (cnt < len && rules[cnt++] == 'S');
	while (cnt < len && isdigit(rules[cnt])) survivalRules |= 1<<(rules[cnt++]-'0');
	assert (cnt == len);
}

std::string CellularAutomaton::getRules() {
	std::ostringstream sstream;
	sstream<<"B";
	for(unsigned i=0; i < 10 && (birthRules>>i) != 0; ++i)
		if (((birthRules>>i)&1) != 0) sstream<<i;
	sstream<<"/S";
	for(unsigned i=0; i < 10 && (survivalRules>>i) != 0; ++i)
		if (((survivalRules>>i)&1) != 0) sstream<<i;
	return sstream.str();
}

/* Implementation of clear() and randomize() methods */
void CellularAutomaton::clear() {
	for (auto y = 0; y < height; ++y)
		for (auto x = 0; x < width; ++x)
			setCellState(x, y, false);
}

void CellularAutomaton::randomize(float ratio) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::bernoulli_distribution dist(ratio);
	for (auto y = 0; y < height; ++y)
		for (auto x = 0; x < width; ++x)
			setCellState(x, y, dist(gen));
}

/* Count alive neighbours of a given cell*/
unsigned CellularAutomaton::countNeighbours(unsigned x, unsigned y) {
	unsigned rc = 0;
	for(int dx=-1; dx<2; ++dx)
		for (int dy=-1; dy < 2; ++dy) {
			if (dx == 0 && dy == 0) continue;
			if (isCellAlive(x+dx,y+dy)) ++rc;
		}
	return rc;
}


class CellularAutomatonImpl: public CellularAutomaton {
public:
	CellularAutomatonImpl();

	~CellularAutomatonImpl() { cleanup(); }
	void resize(unsigned w, unsigned h);
	bool step();

	void setCellState(unsigned x, unsigned y, bool s);
	bool isCellAlive(int x, int y);

private:
	/* Helper routines */
	inline bool willSurvive(unsigned neigh) { return ((survivalRules>>neigh)&1)==1; }
	inline bool willBorn(unsigned neigh) { return ((birthRules>>neigh)&1)==1; }
	void cleanup();
	void realloc();
	void swap();
	void setCellPassiveState(unsigned x, unsigned y, bool s);
	/* Active and passive sets */
	bool *activeSet,*passiveSet;
};

CellularAutomaton *createAutomatonCPU() {
	return new CellularAutomatonImpl();
}


/* Implementation of CellularlAutomatonImpl class */
CellularAutomatonImpl::CellularAutomatonImpl():CellularAutomaton(), activeSet(NULL),passiveSet(NULL) {}

inline void CellularAutomatonImpl::setCellPassiveState(unsigned x, unsigned y, bool s) {
	passiveSet[y*width+x] = s;
}


void CellularAutomatonImpl::resize(unsigned w, unsigned h) {
	CellularAutomaton::resize(w,h);
	realloc();
}

inline void CellularAutomatonImpl::setCellState(unsigned x, unsigned y, bool s) {
	activeSet[y*width+x] = s;
}

inline bool CellularAutomatonImpl::isCellAlive(int x, int y) {
	if (x < 0 || x >= width || y < 0 || y >= height) return false;
	return activeSet[y*width+x];
}


bool CellularAutomatonImpl::step() {
	if (width == 0 || height == 0) return false;
	for (auto y = 0; y < height; y++)
		for (auto x = 0; x < width; x++) {
			bool alive = isCellAlive(x,y);
			unsigned neigh = countNeighbours(x, y);
			setCellPassiveState(x,y, alive? willSurvive(neigh) : willBorn(neigh));
		}

	swap();
	return true;
}

void CellularAutomatonImpl::swap() {
	bool *tmp = activeSet;
	activeSet = passiveSet;
	passiveSet = tmp;
}

template<typename T> static inline void safeDelete(T*& s) {
	if (s == NULL) return;
	delete[] s;
	s = NULL;
}

void CellularAutomatonImpl::cleanup() {
	safeDelete (activeSet);
	safeDelete (passiveSet);
}

void CellularAutomatonImpl::realloc() {
	cleanup();

	activeSet = new bool[width*height];
	passiveSet = new bool[width*height];
}

