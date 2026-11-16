// q2_v3_tree.cu -- x^T A y, version 3 (coalesced + tree reduction).
// Same coalesced accumulation as version 2.  The partial sums are then
// reduced on the GPU with a binary tree inside each thread block
// (log2(blocksize) halving steps separated by __syncthreads()), working on
// a scratch array in GLOBAL memory.  Thread 0 of each block adds the block
// total to the result with one atomicAdd, so only 4 bytes are copied back
// and the host does no merging.
#define VERSION "v3_tree"
#include "q2_common.h"

static float *d_part, *d_result;

__global__ void xtay_tree(const float *A, const float *x, const float *y, int logn, size_t n,
			  size_t total, float *part, float *result)
{
	size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = (size_t)gridDim.x * blockDim.x;
	float s = 0;
	for (size_t e = id; e < total; e += stride) s += x[e >> logn] * A[e] * y[e & (n - 1)];

	float *bp = part + (size_t)blockIdx.x * blockDim.x; // this block's slice
	bp[threadIdx.x] = s;
	__syncthreads();
	for (unsigned h = blockDim.x / 2; h > 0; h >>= 1) {
		if (threadIdx.x < h) bp[threadIdx.x] += bp[threadIdx.x + h];
		__syncthreads();
	}
	if (threadIdx.x == 0) atomicAdd(result, bp[0]);
}

static void version_setup(size_t m, size_t n, long long t, int bs)
{
	CUDA_CHECK(cudaMalloc(&d_part, sizeof(float) * t));
	CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
}

static float version_run(const float *A, const float *x, const float *y, size_t m, size_t n, long long t, int bs)
{
	float r;
	CUDA_CHECK(cudaMemsetAsync(d_result, 0, sizeof(float)));
	xtay_tree<<<(unsigned)(t / bs), bs>>>(A, x, y, log2_exact(n), n, m * n, d_part, d_result);
	CUDA_CHECK(cudaMemcpy(&r, d_result, sizeof(float), cudaMemcpyDeviceToHost));
	return r;
}

static void version_cleanup(void) { cudaFree(d_part); cudaFree(d_result); }
