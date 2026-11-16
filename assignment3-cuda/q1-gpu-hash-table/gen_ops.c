// gen_ops.c -- operation-list generator for Question 1 (programs P1 and P2).
//
// Usage: ./gen_ops <outfile> <nops> <ins%> <del%> <srch%> [N] [seed]
//
//   nops         total number of operations in the list
//   ins/del/srch percentages, must add up to 100
//   N            keys are drawn from [0, N)        (default 2^30)
//   seed         srandom() seed                    (default 12345)
//
// Properties of the generated list:
//   - Operations are segregated: all inserts first, then all deletes,
//     then all searches.
//   - Insert keys are random() % N (duplicates allowed).
//   - Every delete and every search is on a key that has been inserted
//     (positive queries).  Delete keys are distinct (otherwise a second
//     delete of the same key would return 0).  Search keys are drawn only
//     from inserted keys that are NOT deleted, so every search returns 1.
//
// File format (binary, little endian): nops records of two int32 values
//   { op, key }   with op = 0 (insert), 1 (delete), 2 (search).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OP_INSERT 0
#define OP_DELETE 1
#define OP_SEARCH 2

static int cmp_int(const void *a, const void *b)
{
	int x = *(const int *)a, y = *(const int *)b;
	return (x > y) - (x < y);
}

static int cmp_ll(const void *a, const void *b)
{
	long long x = *(const long long *)a, y = *(const long long *)b;
	return (x > y) - (x < y);
}

// 62-bit random number from two random() calls (random() gives 31 bits)
static long long rand62(void)
{
	return ((long long)random() << 31) | (long long)random();
}

int main(int argc, char *argv[])
{
	if (argc < 6) {
		fprintf(stderr, "Usage: %s <outfile> <nops> <ins%%> <del%%> <srch%%> [N] [seed]\n", argv[0]);
		return 1;
	}
	const char *fname = argv[1];
	long long nops = atoll(argv[2]);
	int pi = atoi(argv[3]), pd = atoi(argv[4]), ps = atoi(argv[5]);
	long long N = (argc > 6) ? atoll(argv[6]) : (1LL << 30);
	unsigned seed = (argc > 7) ? (unsigned)atoi(argv[7]) : 12345u;

	if (pi + pd + ps != 100 || pi <= 0 || pd < 0 || ps < 0) {
		fprintf(stderr, "Percentages must be non-negative, add up to 100, and include some inserts\n");
		return 1;
	}
	if (N <= 0 || N > 2147483647LL) {
		fprintf(stderr, "N must be in [1, 2^31-1]\n");
		return 1;
	}
	srandom(seed);

	long long ni = nops * pi / 100;
	long long nd = nops * pd / 100;
	long long ns = nops - ni - nd;

	int *ins = (int *)malloc(sizeof(int) * ni);
	int *uniq = (int *)malloc(sizeof(int) * ni);
	if (!ins || !uniq) { fprintf(stderr, "Out of memory\n"); return 1; }

	for (long long i = 0; i < ni; i++) ins[i] = (int)(random() % N);

	// Distinct inserted keys.  Every delete needs its own distinct present
	// key, so if the random inserts contain too few distinct keys (e.g. the
	// 50% insert / 50% delete mix needs ALL inserts distinct), redraw just
	// enough duplicate occurrences until #distinct >= #deletes.
	long long nu = 0;
	if (nd > ni) { fprintf(stderr, "More deletes than inserts\n"); return 1; }
	for (int round = 0; ; round++) {
		memcpy(uniq, ins, sizeof(int) * ni);
		qsort(uniq, ni, sizeof(int), cmp_int);
		nu = 0;
		for (long long i = 0; i < ni; i++)
			if (i == 0 || uniq[i] != uniq[i - 1]) uniq[nu++] = uniq[i];
		if (nu >= nd) break;
		if (round == 50 || N < nd) {
			fprintf(stderr, "Cannot get %lld distinct keys from [0, %lld); increase N\n", nd, N);
			return 1;
		}
		// Find repeated occurrences: sort (key, position) pairs
		long long *kp = (long long *)malloc(sizeof(long long) * ni);
		if (!kp) { fprintf(stderr, "Out of memory\n"); return 1; }
		for (long long i = 0; i < ni; i++) kp[i] = ((long long)ins[i] << 32) | i;
		qsort(kp, ni, sizeof(long long), cmp_ll);
		long long need = nd - nu;
		for (long long i = 1; i < ni && need > 0; i++)
			if ((kp[i] >> 32) == (kp[i - 1] >> 32)) {
				ins[kp[i] & 0xffffffffLL] = (int)(random() % N);
				need--;
			}
		free(kp);
	}
	if (ns > 0 && nu - nd <= 0) {
		fprintf(stderr, "No surviving keys to search\n");
		return 1;
	}

	// Fisher-Yates shuffle of the distinct keys: the first nd become the
	// delete keys, the rest are the pool for searches.
	for (long long i = nu - 1; i > 0; i--) {
		long long j = rand62() % (i + 1);
		int t = uniq[i]; uniq[i] = uniq[j]; uniq[j] = t;
	}

	FILE *fp = fopen(fname, "wb");
	if (!fp) { perror("fopen"); return 1; }

	enum { BUF = 1 << 16 };
	static int buf[2 * BUF];
	int nb = 0;
#define EMIT(o, k) do { buf[2*nb] = (o); buf[2*nb+1] = (k); \
		if (++nb == BUF) { fwrite(buf, sizeof(int), 2*nb, fp); nb = 0; } } while (0)

	for (long long i = 0; i < ni; i++) EMIT(OP_INSERT, ins[i]);
	for (long long i = 0; i < nd; i++) EMIT(OP_DELETE, uniq[i]);
	for (long long i = 0; i < ns; i++) EMIT(OP_SEARCH, uniq[nd + rand62() % (nu - nd)]);
	if (nb) fwrite(buf, sizeof(int), 2 * nb, fp);
	fclose(fp);

	printf("Wrote %s: %lld ops (%lld insert, %lld delete, %lld search), %lld distinct insert keys, N=%lld\n",
	       fname, nops, ni, nd, ns, nu, N);
	free(ins);
	free(uniq);
	return 0;
}
