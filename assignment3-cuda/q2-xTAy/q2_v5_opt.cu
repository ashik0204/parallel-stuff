// q2_v5_opt.cu -- x^T A y, version 5 (additional optimisations on top of v4).
//   1. Vectorised loads: A and y are read as float4 (16 bytes per thread per
//      load, a warp moves 512 bytes per instruction), quartering the number
//      of load instructions and loop iterations.  Needs n % 4 == 0, i.e. n >= 4
//      (for n < 4 the scalar kernel xtay_scalar below is used instead).
//   2. Read-only data path (__ldg / const __restrict__) for A, x, y.
//   3. Two independent accumulators per thread (loop unrolled by 2) to
//      keep more loads in flight (memory-level parallelism).
//   4. Reduction: warp-level tree with __shfl_down_sync (registers only,
//      no shared memory traffic or __syncthreads), then one shared-memory
//      slot per warp, then the first warp reduces those with shuffles.
// If t exceeds the number of float4 groups (m*n/4) the extra threads
// simply have no elements and contribute 0.
#define VERSION "v5_opt"
#include "q2_common.h"

static float *d_result;

__device__ __forceinline__ float warp_sum(float v)
{
	for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
	return v;
}

__global__ void xtay_opt(const float4 *__restrict__ A4, const float *__restrict__ x,
			 const float4 *__restrict__ y4, int logn4, size_t n4, size_t total4,
			 float *result)
{
	__shared__ float warp_part[32];
	size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = (size_t)gridDim.x * blockDim.x;
	float s0 = 0, s1 = 0;

	size_t e = id;
	for (; e + stride < total4; e += 2 * stride) {
		float4 a0 = __ldg(&A4[e]), a1 = __ldg(&A4[e + stride]);
		float4 b0 = __ldg(&y4[e & (n4 - 1)]), b1 = __ldg(&y4[(e + stride) & (n4 - 1)]);
		float x0 = __ldg(&x[e >> logn4]), x1 = __ldg(&x[(e + stride) >> logn4]);
		s0 += x0 * (a0.x * b0.x + a0.y * b0.y + a0.z * b0.z + a0.w * b0.w);
		s1 += x1 * (a1.x * b1.x + a1.y * b1.y + a1.z * b1.z + a1.w * b1.w);
	}
	if (e < total4) {
		float4 a0 = __ldg(&A4[e]);
		float4 b0 = __ldg(&y4[e & (n4 - 1)]);
		s0 += __ldg(&x[e >> logn4]) * (a0.x * b0.x + a0.y * b0.y + a0.z * b0.z + a0.w * b0.w);
	}
	if (blockDim.x < 32) { // tiny t: partial warp, full-mask shuffle not allowed
		atomicAdd(result, s0 + s1);
		return;
	}
	float s = warp_sum(s0 + s1);

	int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	if (lane == 0) warp_part[warp] = s;
	__syncthreads();
	if (warp == 0) {
		int nwarps = (blockDim.x + 31) >> 5;
		s = (lane < nwarps) ? warp_part[lane] : 0.0f;
		s = warp_sum(s);
		if (lane == 0) atomicAdd(result, s);
	}
}

// Fallback for n < 4 (rows shorter than one float4): scalar coalesced loop,
// same reduction.
__global__ void xtay_scalar(const float *A, const float *x, const float *y, int logn, size_t n,
			    size_t total, float *result)
{
	__shared__ float warp_part[32];
	size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = (size_t)gridDim.x * blockDim.x;
	float s = 0;
	for (size_t e = id; e < total; e += stride) s += x[e >> logn] * A[e] * y[e & (n - 1)];
	if (blockDim.x < 32) { atomicAdd(result, s); return; }
	s = warp_sum(s);
	int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	if (lane == 0) warp_part[warp] = s;
	__syncthreads();
	if (warp == 0) {
		s = (lane < (int)((blockDim.x + 31) >> 5)) ? warp_part[lane] : 0.0f;
		s = warp_sum(s);
		if (lane == 0) atomicAdd(result, s);
	}
}

static void version_setup(size_t m, size_t n, long long t, int bs)
{
	CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
}

static float version_run(const float *A, const float *x, const float *y, size_t m, size_t n, long long t, int bs)
{
	float r;
	size_t n4 = n / 4;
	CUDA_CHECK(cudaMemsetAsync(d_result, 0, sizeof(float)));
	if (n < 4) {
		xtay_scalar<<<(unsigned)(t / bs), bs>>>(A, x, y, log2_exact(n), n, m * n, d_result);
		CUDA_CHECK(cudaMemcpy(&r, d_result, sizeof(float), cudaMemcpyDeviceToHost));
		return r;
	}
	xtay_opt<<<(unsigned)(t / bs), bs>>>((const float4 *)A, x, (const float4 *)y, log2_exact(n4), n4,
					      m * n4, d_result);
	CUDA_CHECK(cudaMemcpy(&r, d_result, sizeof(float), cudaMemcpyDeviceToHost));
	return r;
}

static void version_cleanup(void) { cudaFree(d_result); }
