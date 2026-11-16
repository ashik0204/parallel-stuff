// q2_v2_coalesced.cu -- x^T A y, version 2 (basic + memory access coalescing).
// Thread id handles elements id, id+t, id+2t, ...  In every loop iteration
// the 32 threads of a warp read 32 consecutive floats of A (one 128-byte
// block) and consecutive y[j], so each warp load is a single coalesced
// transaction.  Partial sums are still merged on the host as in version 1.
#define VERSION "v2_coalesced"
#include "q2_common.h"

static float *d_part, *h_part;

__global__ void xtay_coalesced(const float *A, const float *x, const float *y, int logn, size_t n,
			       size_t total, float *part)
{
	size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = (size_t)gridDim.x * blockDim.x;
	float s = 0;
	for (size_t e = id; e < total; e += stride) s += x[e >> logn] * A[e] * y[e & (n - 1)];
	part[id] = s;
}

static void version_setup(size_t m, size_t n, long long t, int bs)
{
	CUDA_CHECK(cudaMalloc(&d_part, sizeof(float) * t));
	CUDA_CHECK(cudaMallocHost(&h_part, sizeof(float) * t));
}

static float version_run(const float *A, const float *x, const float *y, size_t m, size_t n, long long t, int bs)
{
	xtay_coalesced<<<(unsigned)(t / bs), bs>>>(A, x, y, log2_exact(n), n, m * n, d_part);
	CUDA_CHECK(cudaMemcpy(h_part, d_part, sizeof(float) * t, cudaMemcpyDeviceToHost));
	double s = 0;
	for (long long i = 0; i < t; i++) s += h_part[i];
	return (float)s;
}

static void version_cleanup(void) { cudaFree(d_part); cudaFreeHost(h_part); }
