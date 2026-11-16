# Builds for the GPU of the machine it is compiled on (-arch=native, CUDA >= 11.6).
# With an older nvcc, native is not available and nvcc's default architecture
# is used instead.  Override explicitly with e.g.  make ARCH=sm_86  /  sm_61 / sm_30.
NVCC ?= nvcc
ARCH ?= $(shell $(NVCC) -arch=native -x cu -E /dev/null >/dev/null 2>&1 && echo native)
ARCHFLAG = $(if $(ARCH),-arch=$(ARCH),)
NVFLAGS = -O3 $(ARCHFLAG) -std=c++14

all: gen_ops p1 p2

gen_ops: gen_ops.c
	gcc -O2 -o $@ $<

p1: p1.cu q1_common.h
	$(NVCC) $(NVFLAGS) -o $@ p1.cu

p2: p2.cu q1_common.h
	$(NVCC) $(NVFLAGS) -o $@ p2.cu

clean:
	rm -f gen_ops p1 p2

.PHONY: all clean
