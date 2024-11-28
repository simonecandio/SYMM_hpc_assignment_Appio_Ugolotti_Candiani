# Disable built-in Makefile rules to speed up the build process
MAKEFLAGS+= -r

# Extend CXXFLAGS (compiler flags for C++) with additional parameters provided at the command line
CXXFLAGS += -O2 ${EXTRA_CXXFLAGS}

# Specify the location of the local CUDA installation

# CUDA_HOME := /usr/local/cuda-10.0
CUDA_HOME := /usr/local/cuda

# Define the path to the NVIDIA CUDA compiler (nvcc)
NVCC := $(CUDA_HOME)/bin/nvcc

# Specify the C++ compiler
CXX := g++

# Define the flags to be used by the nvcc compiler
NVCFLAGS := $(CXXFLAGS) $(NVOPT)

# Name of the target executable to be generated
EXE := symm.exe

# Name of the source file containing the CUDA code
SRC := symm.cu

# Define the corresponding object file name (same base name as the source file, with a .o extension)
OBJ := $(SRC:.cu=.o)

# Default target that builds the executable
all: $(EXE)

# Rule to link the object file and create the executable
$(EXE): $(OBJ)
        $(NVCC) $(NVCFLAGS) $(OBJ) -o $@

# Rule to compile the CUDA source file into an object file
$(OBJ): $(SRC)
        $(NVCC) $(NVCFLAGS) -c $< -o $@

# Target to run the compiled executable
# .PHONY indicates this target is not associated with any file
.PHONY: run
run: $(EXE)
        ./$(EXE)

# Target to clean up generated files (object files and executable)
clean:
        rm -f *.o $(EXE)

