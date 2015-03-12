#include "CellularAutomatonCUDA.h"
#include <iostream>

namespace CUCA {
/* Global variables containing automata rules */
__device__ unsigned birthRules;
__device__ unsigned survivalRules;

/* Helper functions */
__device__ inline bool willSurvive(unsigned neigh) {
	return ((survivalRules>>neigh)&1) == 1;
}

__device__ inline bool willBorn(unsigned neigh) {
	return ((birthRules>>neigh)&1) == 1;
}

__device__ inline bool checkBounds(int offs, int size) {
	return (offs >= 0 && offs <= size);
}

__device__ inline unsigned countNeighbours(unsigned x, unsigned y, bool *set, int width, int height) {
	unsigned rc=0;
	int offs = y*width + x;
	for (int yoffs = -width; yoffs <= width; yoffs += width)
		for(int xoffs = -1; xoffs <= 1;) {
			int noffs = offs+yoffs+xoffs;
			if (noffs == offs) continue;
			if (!checkBounds(noffs, width*height)) continue;
			if (set[noffs]) rc++;
		}
	return rc;
}

/* step kernel to be invoked for every cell in the grid */
__global__ void step(bool *inSet, bool *outSet, int width, int height) {
	unsigned x = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned y = blockIdx.y*blockDim.y+threadIdx.y;
	int offs = y*width + x;

	if (!checkBounds(offs, width*height)) return;

	bool val = inSet[offs];
	unsigned neigh = countNeighbours(x,y, inSet, width, height);
	outSet[offs] = val ? willSurvive(neigh) : willBorn(neigh);
}

__global__ void setRules(unsigned b, unsigned s) {
	birthRules = b;
	survivalRules = s;
}

} /* end of CUCA namespace*/


inline int divRoundUp(int x, int y) { return (x+y-1)/y; }
/* step kernel invocation from the host */
bool CellularAutomatonCUDA::step() {
	cudaError_t rc;
	/* Transfer memory from the host */
	if (!syncMemory())
		return false;

	/* Calculate kernel parameters */
	const int blockWidth = 16;
	dim3 blockSize(blockWidth,blockWidth);
	dim3 numBlocks(divRoundUp(width,blockWidth), divRoundUp(height,blockWidth));
	/* Launch the kernel */
	CUCA::step<<<numBlocks,blockSize>>>(gpuActiveSet, gpuPassiveSet, width, height);
	rc = cudaGetLastError();
	if (rc != cudaSuccess) {
		std::cout <<"CallAutomationEvaluateStep launch failed "<<rc<<std::endl;
		return false;
	}
	/* Wait for the kernel to complete */
	rc = cudaDeviceSynchronize();
	if (rc != cudaSuccess) {
		std::cout <<"cudaDeviceSynchronize()="<<rc<<std::endl;
		return false;
	}
	swap();
	needUpdateCPU = true;
	return true;
}


inline void CellularAutomatonCUDA::updateRulesOnGPU() {
	CUCA::setRules<<<1,1>>>(birthRules, survivalRules);
}


CellularAutomatonCUDA::CellularAutomatonCUDA():CellularAutomaton(), needUpdateCPU(false),needUpdateGPU(false),gpuActiveSet(NULL), gpuPassiveSet(NULL), cpuSet(NULL) {
	updateRulesOnGPU();
}

CellularAutomatonCUDA::~CellularAutomatonCUDA() {
	cleanup();
}

template<typename T> void cudaArrayAlloc(T *&ptr, unsigned nElements) {
	cudaError_t rc = cudaMalloc (&ptr, sizeof(T)*nElements);
	if (rc != cudaSuccess)
		std::cout <<"cudaMalloc(&ptr, "<<(sizeof(T)*nElements)<<")="<<rc<<std::endl;
}

void CellularAutomatonCUDA::resize(unsigned w, unsigned h) {
	CellularAutomaton::resize(w,h);
	cleanup();
	cpuSet = new bool[width*height];
	cudaArrayAlloc (gpuActiveSet, width*height);
	cudaArrayAlloc (gpuPassiveSet, width*height);
	needUpdateCPU = needUpdateGPU = false;
}

void CellularAutomatonCUDA::setRules(const std::string &s) {
	CellularAutomaton::setRules(s);
	updateRulesOnGPU();
}

inline void CellularAutomatonCUDA::setCellState(unsigned x, unsigned y, bool s) {
	cpuSet[y*width+x] = s;
	needUpdateGPU = true;
}

inline bool CellularAutomatonCUDA::syncMemory() {
	cudaError_t rc;
	if (needUpdateCPU) {
		rc = cudaMemcpy (cpuSet, gpuActiveSet, sizeof(bool)*width*height, cudaMemcpyDeviceToHost);
		if (rc != cudaSuccess) {
			std::cout <<"cudaMemcpy()="<<rc<<std::endl;
			return false;
		}
		needUpdateCPU = false;
	}
	if (needUpdateGPU) {
		rc = cudaMemcpy (gpuActiveSet, cpuSet, sizeof(bool)*width*height, cudaMemcpyHostToDevice);
		if (rc != cudaSuccess) {
			std::cout <<"cudaMemcpy()="<<rc<<std::endl;
			return false;
		}
		needUpdateGPU = false;
	}
	return true;
}

inline bool CellularAutomatonCUDA::isCellAlive(int x, int y) {
	if (x < 0 || x >= width || y < 0 || y >= height) return false;
	syncMemory();
	return cpuSet[y*width+x];
}

void CellularAutomatonCUDA::swap() {
	bool *tmp = gpuActiveSet;
	gpuActiveSet = gpuPassiveSet;
	gpuPassiveSet = tmp;
}

template<typename T> static void cudaSafeFreeClean(T *& ptr) {
	if (ptr == NULL) return;

	cudaFree (ptr);
	ptr = NULL;
}

void CellularAutomatonCUDA::cleanup() {
	if (cpuSet) {
		delete[] cpuSet;
		cpuSet = NULL;
	}

	cudaSafeFreeClean (gpuActiveSet);
	cudaSafeFreeClean (gpuPassiveSet);
}

std::ostream& operator<<(std::ostream &os, const cudaError_t err) {
	return os<<(unsigned)err<<"("<<cudaGetErrorString(err)<<")";
}

CellularAutomaton *createAutomatonCUDA() {
	return new CellularAutomatonCUDA();
}
