OS=$(shell uname)
ARCH=$(shell uname -m)

TARGET=GameOfLife
OBJS=main.o utils.o GLUTWrapper.o CellularAutomaton.o CellularAutomatonCUDA.o AutomatonViewer.o

CXXFLAGS += -std=c++11 -Wall -Iinc/ -Wno-deprecated-declarations

ifeq ($(DBG),1)
  CXXFLAGS += -g
else
  CXXFLAGS += -O3
endif

# Specify link rules for OpenGL/GLUT
ifeq ($(OS),Darwin)
  FRAMEWORKS=OpenGL GLUT
  LDFLAGS +=$(foreach fw,$(FRAMEWORKS), -framework $(fw))
endif

ifeq ($(OS),Linux)
  LDFLAGS +=-lGL -lglut -lpthread
endif

# CUDA specific defines
CUDA_BASEDIR ?= /usr/local/cuda

CUDA_BINDIR=$(CUDA_BASEDIR)/bin
ifeq ($(OS), Linux)
  ifeq ($(ARCH),x86_64)
    CUDA_LIBDIR = $(CUDA_BASEDIR)/lib64
  endif
endif


CUDA_LIBDIR ?= $(CUDA_BASEDIR)/lib
NVCC=$(CUDA_BINDIR)/nvcc
NVCC_FLAGS += -Iinc/ -arch=compute_30
ifeq ($(OS),Linux)
  NVCC_FLAGS += -Xcompiler -Wall
endif
ifeq ($(DBG),1)
  NVCC_FLAGS += -g -G
else
  NVCC_FLAGS += -O3 -lineinfo
endif

LDFLAGS += $(CUDA_LIBDIR)/libcudart_static.a -ldl -lpthread
#shm_open and friends are located in librt on Linux
ifeq ($(OS),Linux)
  LDFLAGS += -lrt
endif
ifeq ($(OS),Darwin)
  FRAMEWORKS += CUDA
endif



all: obj $(TARGET)

run: obj $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(TARGET) obj

obj:
	mkdir obj

$(TARGET): $(foreach obj, $(OBJS), obj/$(obj))
	$(CXX) -o $@ $^ $(LDFLAGS)

obj/%.o: src/%.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

obj/%.o: src/%.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<
