#ifndef __CELLULAR_AUTOMATON_CUDA_H__
#define __CELLULAR_AUTOMATON_CUDA_H__

#include "CellularAutomaton.h"

class CellularAutomatonCUDA: public CellularAutomaton {
public:
	CellularAutomatonCUDA();
	~CellularAutomatonCUDA();

	void resize(unsigned w, unsigned h);
	void setRules(const std::string &s);
	void setCellState(unsigned x, unsigned y, bool s);
	bool isCellAlive(int x, int y);
	bool step();

private:
	inline void updateRulesOnGPU();
	inline bool syncMemory();
	void cleanup();
	void swap();

	bool needUpdateCPU;
	bool needUpdateGPU;
	bool *gpuActiveSet, *gpuPassiveSet;
	bool *cpuSet;
};

#endif /*__CELLULAR_AUTOMATON_CUDA_H__*/
