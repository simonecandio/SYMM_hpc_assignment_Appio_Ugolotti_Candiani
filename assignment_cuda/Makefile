-include ../../../utilities/options.mk
-include ../../../utilities/c2.mk
MAKEFLAGS+= --silent
MAKEFLAGS+= -r
# Extend CFLAGS with command line parameters
#CXXFLAGS+= -DPOLYBENCH_TIME
CXXFLAGS += -O2 ${EXTRA_CXXFLAGS}
 
CXXFLAGS += $(EXT_CFLAGS)

# Select the location of the local CUDA install
# CUDA_HOME := /usr/local/cuda-10.0
CUDA_HOME:=/usr/local/cuda

# Specify the directory of the nvcc compiler
NVCC:=$(CUDA_HOME)/bin/nvcc
CXX := g++

# Specify the flags for the nvcc compiler
NVCFLAGS := $(CXXFLAGS) $(NVOPT)


# Target executable
EXE := symm.exe

# Source file
SRC := symm.cu

# Object file
OBJ := $(SRC:.cu=.o)

# Default target
all: $(EXE)

# Rule to build the executable
$(EXE): $(OBJ)
	$(NVCC) $(NVCFLAGS) $(OBJ) -o $@

# Rule to compile the CUDA source into an object file
$(OBJ): $(SRC)
	$(NVCC) $(NVCFLAGS) -c $< -o $@



.PHONY: run profile clean
run: $(EXE)
	./$(EXE)

profile: $(EXE)
	sudo $(CUDA_HOME)/bin/nvprof --unified-memory-profiling off ./$(EXE)

metrics: $(EXE)
	sudo $(CUDA_HOME)/bin/nvprof --print-gpu-trace --metrics "eligible_warps_per_cycle,achieved_occupancy,sm_efficiency,ipc" ./$(EXE)
        
# Clean up generated files
clean:
	rm -f *.o $(EXE)
