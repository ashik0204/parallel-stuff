// q2_v4_shmem.cu -- x^T A y, version 4 (coalesced + tree reduction in
// shared memory).
// Identical to version 3 except that the per-block tree reduction runs on
// a __shared__ array (on-chip memory of the SM, with much lower latency than
// global memory and no L2/DRAM traffic), and the global scratch array disappears.
#define VERSION "v4_shmem"
#include "q2_common.h"

static float *d_result;

__global__ void xtay_shmem(const float *A, const float *x, const float *y, int logn, size_t n,
			   size_t total, float *result)
{
	extern __shared__ float sm[];
	size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = (size_t)gridDim.x * blockDim.x;
	float s = 0;
	for (size_t e = id; e < total; e += stride) s += x[e >> logn] * A[e] * y[e & (n - 1)];

	sm[threadIdx.x] = s;
	__syncthreads();
	for (unsigned h = blockDim.x / 2; h > 0; h >>= 1) {
		if (threadIdx.x < h) sm[threadIdx.x] += sm[threadIdx.x + h];
		__syncthreads();
	}
	if (threadIdx.x == 0) atomicAdd(result, sm[0]);
}

static void version_setup(size_t m, size_t n, long long t, int bs)
{
	CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
}

static float version_run(const float *A, const float *x, const float *y, size_t m, size_t n, long long t, int bs)
{
	float r;
	CUDA_CHECK(cudaMemsetAsync(d_result, 0, sizeof(float)));
	xtay_shmem<<<(unsigned)(t / bs), bs, bs * sizeof(float)>>>(A, x, y, log2_exact(n), n, m * n, d_result);
	CUDA_CHECK(cudaMemcpy(&r, d_result, sizeof(float), cudaMemcpyDeviceToHost));
	return r;
}

static void version_cleanup(void) { cudaFree(d_result); }
