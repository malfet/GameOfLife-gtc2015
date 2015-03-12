#include "GLUTWrapper.h"
#include "utils.h"
#include "CellularAutomaton.h"
#include "AutomatonViewer.h"
#include <chrono>
#include <sstream>
#include <iostream>

int profileRun(CellularAutomaton *automaton, unsigned nSteps=24) {
	using namespace std::chrono;
	DeleteMe<CellularAutomaton> cleanup(automaton);

	automaton->resize(512,512);
	automaton->randomize();

	/* Warm up */
	if (!automaton->step())
		return -1;

	/* Benchmark */
	auto start = steady_clock::now();

	for(unsigned cnt=0;cnt<nSteps;cnt++)
		if (!automaton->step()) break;

	auto stop = steady_clock::now();
	auto duration = getDuration<microseconds>(stop-start);

	std::cout<<nSteps<<" steps took "<<duration<<" microseconds"<<std::endl;

	return 0;
}

int startViewer(int *argc, char *argv[], CellularAutomaton *a) {
	GLUTWrapper *wrapper = GLUTWrapper::create(argc, argv);
	AutomatonViewer viewer(wrapper, a);
	wrapper->run();
	return 0;
}

CellularAutomaton *createAutomaton(bool cpu=false) {
	return cpu ? createAutomatonCPU() : createAutomatonCUDA();
}

int main(int argc, char *argv[]) {
	bool onCPU = hasOption(argc, argv, "--cpu");
	CellularAutomaton *a = createAutomaton(onCPU);

	if (hasOption(argc, argv, "--profile"))
		return profileRun(a);

	return startViewer(&argc, argv, a);
}
