#include "CellularAutomatonCUDA.h"
#include <iostream>

namespace CUCA {
/* Global variables containing automata rules */
__device__ unsigned birthRules;
__device__ unsigned survivalRules;

/* Slice of the grid shared between threads withing the block*/
extern __shared__ bool sharedSet[];

/* Helper functions */
__device__ inline bool willSurvive(unsigned neigh) {
	return ((survivalRules>>neigh)&1) == 1;
}

__device__ inline bool willBorn(unsigned neigh) {
	return ((birthRules>>neigh)&1) == 1;
}

__device__ inline bool checkBounds(int offs, int size) {
	return (offs >= 0 && offs < size);
}

__device__ inline unsigned countNeighbours() {
	int width = blockDim.x+2;
	int offsets[8] = {-width-1, -width, -width+1, -1, 1, width-1, width, width+1};
	unsigned rc=0;
	int offs = (threadIdx.y+1)*width + threadIdx.x+1;
	for (unsigned cnt = 0; cnt < 8; ++cnt) {
		int noffs = offs+offsets[cnt];
		if (sharedSet[noffs]) rc++;
	}
	return rc;
}

/* step kernel to be invoked for every cell in the grid */
__global__ void step(bool *inSet, bool *outSet, int width, int height) {
	int blockWidth = blockDim.x+2;
	int blockOffs = (threadIdx.y+1)*blockWidth + threadIdx.x+1;
	unsigned x = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned y = blockIdx.y*blockDim.y+threadIdx.y;
	int offs = y*width + x;

	if (!checkBounds(offs, width*height)) {
		sharedSet[blockOffs] = false;
		return;
	}

	bool val = inSet[offs];
	/* Fill in shared set */
	/* Point itself */
	sharedSet[blockOffs] = val;
	/* Edges */
	if (threadIdx.x == 0) sharedSet[blockOffs-1] = offs > 0 ? inSet[offs-1]: false;
	if (threadIdx.x == blockDim.x-1) sharedSet[blockOffs+1] = offs+1 < width*height ? inSet[offs+1]: false;
	if (threadIdx.y == 0) sharedSet[blockOffs-blockWidth] = offs-width >= 0 ? inSet[offs-width]: false;
	if (threadIdx.y == blockDim.y-1) sharedSet[blockOffs+blockWidth] = offs+width < width*height ? inSet[offs+width]: false;
	/* Corners */
	if (threadIdx.x == 0 && threadIdx.y == 0) sharedSet[0] = offs-width > 0 ? inSet[offs-width-1] : false;
	if (threadIdx.x == blockDim.x-1 && threadIdx.y == 0) sharedSet[blockWidth-1] = offs >= width-1 ? inSet[offs-width+1] : false;
	if (threadIdx.x == 0 && threadIdx.y == blockDim.y-1) sharedSet[blockOffs+blockWidth-1] = offs+width <= width*height? inSet[offs+width-1] : false;
	if (threadIdx.x == blockDim.x-1 && threadIdx.y == blockDim.y-1) sharedSet[blockOffs+blockWidth+1] = offs+width+1 < width*height? inSet[offs+width+1] : false;
	__syncthreads();

	unsigned neigh = countNeighbours();
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
	size_t shmemSize = (blockWidth+2)*(blockWidth+2)*sizeof(bool);
	/* Launch the kernel */
	CUCA::step<<<numBlocks,blockSize, shmemSize>>>(gpuActiveSet, gpuPassiveSet, width, height);
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
