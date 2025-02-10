# Makefile for MPI program

CXX = mpic++
CXXFLAGS = -lm
TARGET = main
SRC = main.cpp

all: clean $(TARGET)

$(TARGET): $(SRC)
	$(CXX) -o $@ $^ $(CXXFLAGS)

run:
	mpirun -np 4 ./$(TARGET)

clean:
	if [ -f $(TARGET) ] ; then rm $(TARGET); fi

.PHONY: clean run
