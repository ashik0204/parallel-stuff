// q2_omp.c -- x^T A y with OpenMP (Question 2, CPU baseline for the CUDA programs).
//
// Usage: ./q2_omp <m> <n> <t>          (REPS env var, default 3)
//
// Same matrix/vector values as the CUDA programs (same hash generator).
// Initialisation is parallel with the same static schedule as the compute
// loop, so each thread first-touches the rows it later reads (NUMA/cache
// friendly) -- it is not timed.
//
// Compute: rows are divided statically among threads; each row is reduced
// with a SIMD-vectorised inner loop (omp simd reduction -> AVX2), and the row
// sum is multiplied by x[i] and added in double:  x^T A y = sum_i x_i * (sum_j A_ij y_j).
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

static inline float gen_val(unsigned long long i, unsigned salt)
{
	unsigned long long z = i + 0x9E3779B97F4A7C15ULL * (salt + 1);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	z ^= z >> 31;
	return (float)((int)(z & 0xFFFF) - 32768) * (1.0f / 32768.0f);
}

static double now_sec(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}

static int is_pow2(unsigned long long v) { return v && !(v & (v - 1)); }

// aligned_alloc needs the size to be a multiple of the alignment: round up to 64 bytes
static float *alloc64(size_t count) { return (float *)aligned_alloc(64, (count * sizeof(float) + 63) & ~(size_t)63); }

int main(int argc, char *argv[])
{
	if (argc < 4) { fprintf(stderr, "Usage: %s <m> <n> <t>\n", argv[0]); return 1; }
	size_t m = strtoull(argv[1], NULL, 10), n = strtoull(argv[2], NULL, 10);
	int t = atoi(argv[3]);
	int reps = getenv("REPS") ? atoi(getenv("REPS")) : 3;
	if (!is_pow2(m) || !is_pow2(n) || !is_pow2(t)) { fprintf(stderr, "m, n, t must be powers of two\n"); return 1; }
	omp_set_num_threads(t);

	float *A = alloc64(m * n), *x = alloc64(m), *y = alloc64(n);
	if (!A || !x || !y) { fprintf(stderr, "Out of memory\n"); return 1; }

#pragma omp parallel for schedule(static)
	for (long long i = 0; i < (long long)m; i++) {
		x[i] = gen_val(i, 1);
		for (size_t j = 0; j < n; j++) A[(size_t)i * n + j] = gen_val((size_t)i * n + j, 3);
	}
	for (size_t j = 0; j < n; j++) y[j] = gen_val(j, 2);

	double result = 0, total_t = 0, best = 1e30;
	for (int r = 0; r < reps; r++) {
		double t0 = now_sec();
		double sum = 0;
#pragma omp parallel for reduction(+:sum) schedule(static)
		for (long long i = 0; i < (long long)m; i++) {
			const float *row = A + (size_t)i * n;
			float rs = 0;
			// only y is promised to be 64-byte aligned: a row is not when n < 16
#pragma omp simd reduction(+:rs) aligned(y : 64)
			for (size_t j = 0; j < n; j++) rs += row[j] * y[j];
			sum += (double)x[i] * rs;
		}
		double t1 = now_sec();
		result = sum;
		total_t += t1 - t0;
		if (t1 - t0 < best) best = t1 - t0;
	}
	double avg_ms = total_t / reps * 1e3;

	// Reference in double, not timed
	double ref = 0;
#pragma omp parallel for reduction(+:ref) schedule(static)
	for (long long i = 0; i < (long long)m; i++) {
		double rs = 0;
		for (size_t j = 0; j < n; j++) rs += (double)A[(size_t)i * n + j] * y[j];
		ref += (double)x[i] * rs;
	}
	double diff = fabs(ref - result);
	double gbps = sizeof(float) * (double)(m * n + m + n) / (avg_ms * 1e-3) / 1e9;

	printf("omp m=%zu n=%zu t=%d result=%.6f ref=%.6f |diff|=%.6e time=%.3f ms (min %.3f, %d reps) bandwidth=%.1f GB/s\n",
	       m, n, t, result, ref, diff, avg_ms, best * 1e3, reps, gbps);
	printf("CSV,omp,%zu,%zu,%d,0,0,%.6f,%.6f,%.6e,%.4f,%.4f,%.2f\n",
	       m, n, t, result, ref, diff, avg_ms, best * 1e3, gbps);
	free(A); free(x); free(y);
	return 0;
}
