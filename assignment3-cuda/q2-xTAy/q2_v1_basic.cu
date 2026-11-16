// q2_v1_basic.cu -- x^T A y, version 1 (basic).
// The m*n elements of A (row-major) are split into t contiguous chunks of
// m*n/t elements; thread id handles elements [id*span, (id+1)*span).
// Each thread computes sum x[i]*A[i][j]*y[j] over its chunk and writes it
// to part[id]; the host copies the t partial sums back and adds them.
// Uncoalesced: consecutive threads of a warp touch addresses span*4 bytes
// apart, so every warp load needs up to 32 separate memory transactions.
#define VERSION "v1_basic"
#include "q2_common.h"

static float *d_part, *h_part;

__global__ void xtay_basic(const float *A, const float *x, const float *y, int logn, size_t n,
			   size_t span, size_t total, float *part)
{
	size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t lo = id * span, hi = lo + span; // span >= 1; threads past m*n get nothing
	if (hi > total) hi = total;
	float s = 0;
	for (size_t e = lo; e < hi; e++) s += x[e >> logn] * A[e] * y[e & (n - 1)];
	part[id] = s;
}

static void version_setup(size_t m, size_t n, long long t, int bs)
{
	CUDA_CHECK(cudaMalloc(&d_part, sizeof(float) * t));
	CUDA_CHECK(cudaMallocHost(&h_part, sizeof(float) * t));
}

static float version_run(const float *A, const float *x, const float *y, size_t m, size_t n, long long t, int bs)
{
	xtay_basic<<<(unsigned)(t / bs), bs>>>(A, x, y, log2_exact(n), n,
				       m * n >= (size_t)t ? m * n / t : 1, m * n, d_part);
	CUDA_CHECK(cudaMemcpy(h_part, d_part, sizeof(float) * t, cudaMemcpyDeviceToHost));
	double s = 0;
	for (long long i = 0; i < t; i++) s += h_part[i];
	return (float)s;
}

static void version_cleanup(void) { cudaFree(d_part); cudaFreeHost(h_part); }
