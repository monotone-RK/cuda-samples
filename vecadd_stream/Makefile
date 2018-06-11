################################################################################
# Makefile for CUDA Sample Program                 Ryohei Kobayashi 2018.06.11 #
################################################################################

# Files
SRCS := main.cu

# Compiler
CXX := nvcc

# Compilation flags
CXXFLAGS := -O3 -g -std=c++11 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -fopenmp

# Target
TARGET := vecadd_stream

# User settings
NUMDATA := 1048576
NUMSTREAM := 4
NUMTRY := 10000

compile:
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

prun:
	srun -p comq -w ppx00 ./$(TARGET) $(NUMDATA) $(NUMSTREAM) $(NUMTRY)

vrun:
	srun -p volta-32 -w ppx2-04 ./$(TARGET) $(NUMDATA) $(NUMSTREAM) $(NUMTRY)

clean:
	rm -f $(TARGET)
