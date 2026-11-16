# Builds for the GPU of the machine it is compiled on (-arch=native, CUDA >= 11.6).
# With an older nvcc, native is not available and nvcc's default architecture
# is used instead.  Override explicitly with e.g.  make ARCH=sm_86  /  sm_61 / sm_30.
NVCC ?= nvcc
ARCH ?= $(shell $(NVCC) -arch=native -x cu -E /dev/null >/dev/null 2>&1 && echo native)
ARCHFLAG = $(if $(ARCH),-arch=$(ARCH),)
NVFLAGS = -O3 $(ARCHFLAG) -Xcompiler -fopenmp -lgomp
CUDA_PROGS = q2_v1_basic q2_v2_coalesced q2_v3_tree q2_v4_shmem q2_v5_opt

all: $(CUDA_PROGS) q2_omp

q2_v%: q2_v%.cu q2_common.h
	$(NVCC) $(NVFLAGS) -o $@ $<

q2_omp: q2_omp.c
	gcc -O3 -march=native -fopenmp -o $@ $< -lm

clean:
	rm -f $(CUDA_PROGS) q2_omp

.PHONY: all clean
