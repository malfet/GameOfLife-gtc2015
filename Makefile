OS=$(shell uname)
ARCH=$(shell uname -m)

TARGET=GameOfLife
OBJS=main.o utils.o GLUTWrapper.o CellularAutomaton.o AutomatonViewer.o

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
