// q2_common.h -- shared host driver for the x^T A y CUDA programs (Question 2).
//
// Each version file (q2_v*.cu) defines
//     #define VERSION "<name>"
//     static void  version_setup(size_t m, size_t n, long long t, int bs);
//     static float version_run(const float *A, const float *x, const float *y,
//                              size_t m, size_t n, long long t, int bs);
//     static void  version_cleanup(void);
// and then includes this file, which supplies main().
//
// Usage: ./q2_vK <m> <n> <t> [blocksize=256]   (m, n, t powers of two)
//   REPS environment variable (default 3): timed repetitions.
//
// A (m x n, row-major), x (m), y (n) are filled by a separate CUDA kernel
// (not timed).  Values come from a hash of the element index, so the CPU
// can regenerate exactly the same floats to compute the reference result
// in double precision without copying A back.
//
// Timed region: from just before the compute kernel launch until the
// final scalar result is available on the host (this includes copying and
// merging per-thread partial sums for the versions that need it).
// The CPU reference and the difference are computed after timing.
#ifndef Q2_COMMON_H
#define Q2_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <omp.h>

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
	fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
	exit(1); } } while (0)

#define DEFAULT_BS 256

// Deterministic pseudo-random value in [-1, 1) for index i, stream salt.
// (splitmix64 finaliser; the result is k/32768 for an integer k, so it is
// exactly representable and identical on CPU and GPU.)
__host__ __device__ __forceinline__ float gen_val(unsigned long long i, unsigned salt)
{
	unsigned long long z = i + 0x9E3779B97F4A7C15ULL * (salt + 1);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	z ^= z >> 31;
	return (float)((int)(z & 0xFFFF) - 32768) * (1.0f / 32768.0f);
}

#define SALT_X 1
#define SALT_Y 2
#define SALT_A 3

__global__ void init_kernel(float *A, float *x, float *y, size_t m, size_t n)
{
	size_t stride = (size_t)gridDim.x * blockDim.x;
	size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	for (size_t e = id; e < m * n; e += stride) A[e] = gen_val(e, SALT_A);
	for (size_t i = id; i < m; i += stride) x[i] = gen_val(i, SALT_X);
	for (size_t j = id; j < n; j += stride) y[j] = gen_val(j, SALT_Y);
}

static int log2_exact(unsigned long long v)
{
	int l = 0;
	while ((1ULL << l) < v) l++;
	return l;
}

static bool is_pow2(unsigned long long v) { return v && !(v & (v - 1)); }

static double now_sec(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}

// CPU reference in double precision (parallelised with OpenMP; not timed)
static double cpu_reference(size_t m, size_t n)
{
	double total = 0;
#pragma omp parallel for reduction(+:total) schedule(static)
	for (long long i = 0; i < (long long)m; i++) {
		double row = 0;
		for (size_t j = 0; j < n; j++)
			row += (double)gen_val((size_t)i * n + j, SALT_A) * (double)gen_val(j, SALT_Y);
		total += (double)gen_val(i, SALT_X) * row;
	}
	return total;
}

static void version_setup(size_t m, size_t n, long long t, int bs);
static float version_run(const float *A, const float *x, const float *y, size_t m, size_t n, long long t, int bs);
static void version_cleanup(void);

int main(int argc, char *argv[])
{
	if (argc < 4) {
		fprintf(stderr, "Usage: %s <m> <n> <t> [blocksize=%d]\n", argv[0], DEFAULT_BS);
		return 1;
	}
	size_t m = strtoull(argv[1], NULL, 10);
	size_t n = strtoull(argv[2], NULL, 10);
	long long t = atoll(argv[3]);
	int bs = (argc > 4) ? atoi(argv[4]) : DEFAULT_BS;
	int reps = getenv("REPS") ? atoi(getenv("REPS")) : 3;

	if (!is_pow2(m) || !is_pow2(n) || !is_pow2(t) || !is_pow2(bs) || bs > 1024) {
		fprintf(stderr, "m, n, t and blocksize must be powers of two (blocksize <= 1024)\n");
		return 1;
	}
	// t > m*n is allowed: the surplus threads simply get no elements.
	if (t < bs) bs = (int)t;

	size_t bytes = sizeof(float) * (m * n + m + n);
	size_t freeb, totalb;
	CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
	if (bytes + (64u << 20) > freeb) {
		fprintf(stderr, "A, x, y need %zu MB, GPU has %zu MB free\n", bytes >> 20, freeb >> 20);
		return 1;
	}

	float *A, *x, *y;
	CUDA_CHECK(cudaMalloc(&A, sizeof(float) * m * n));
	CUDA_CHECK(cudaMalloc(&x, sizeof(float) * m));
	CUDA_CHECK(cudaMalloc(&y, sizeof(float) * n));

	// Initialisation on the GPU (not timed)
	init_kernel<<<1024, 256>>>(A, x, y, m, n);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	version_setup(m, n, t, bs);

	double total = 0, best = 1e30;
	float gpu = 0;
	for (int r = 0; r < reps; r++) {
		CUDA_CHECK(cudaDeviceSynchronize());
		double t0 = now_sec();
		gpu = version_run(A, x, y, m, n, t, bs); // returns after result reaches host
		double t1 = now_sec();
		CUDA_CHECK(cudaGetLastError());
		total += t1 - t0;
		if (t1 - t0 < best) best = t1 - t0;
	}
	double avg_ms = total / reps * 1e3;

	// Not timed: CPU reference and difference
	double ref = cpu_reference(m, n);
	double diff = fabs(ref - (double)gpu);
	double gbps = (double)bytes / (avg_ms * 1e-3) / 1e9;

	printf("%s m=%zu n=%zu t=%lld blocks=%lld blocksize=%d GPU=%.6f CPU=%.6f |diff|=%.6e rel=%.3e "
	       "time=%.3f ms (min %.3f, %d reps) bandwidth=%.1f GB/s\n",
	       VERSION, m, n, t, t / bs, bs, gpu, ref, diff, diff / (fabs(ref) + 1e-30),
	       avg_ms, best * 1e3, reps, gbps);
	printf("CSV,%s,%zu,%zu,%lld,%lld,%d,%.6f,%.6f,%.6e,%.4f,%.4f,%.2f\n",
	       VERSION, m, n, t, t / bs, bs, gpu, ref, diff, avg_ms, best * 1e3, gbps);

	version_cleanup();
	cudaFree(A); cudaFree(x); cudaFree(y);
	return 0;
}

#endif
