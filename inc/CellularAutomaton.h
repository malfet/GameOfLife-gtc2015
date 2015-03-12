#ifndef __CELLULAR_AUTOMATON_H__
#define __CELLULAR_AUTOMATON_H__
#include <stdint.h>
#include <stdbool.h>
#include <string>

/* Abstract automaton class */
class CellularAutomaton {
public:
	CellularAutomaton(const std::string &rules="B3/S23") {setRules(rules);}
	virtual ~CellularAutomaton() {}
	unsigned countNeighbours(unsigned x, unsigned y);
	void clear();
	void randomize(float ratio=.5);
	virtual void setRules(const std::string &);
	std::string getRules();
public:
	/* Grid manipulation methods */
	virtual void setCellState(unsigned x, unsigned y, bool s) {}
	virtual bool isCellAlive(int x, int y) { return false; }
	/**
	 *  step() method returns true if it was successful
	 *         or false, if error have occurred
	 */
	virtual bool step() { return false; }
	virtual void resize(unsigned w, unsigned h) { width = w; height = h; }
protected:
	int width, height;
	unsigned birthRules;
	unsigned survivalRules;
};

CellularAutomaton *createAutomaton();


#endif /*__CELLULAR_AUTOMATON_H__*/
