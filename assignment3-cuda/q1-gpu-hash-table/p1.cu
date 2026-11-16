// p1.cu -- Question 1, Program P1.
// Concurrent open-addressed hash table: each thread executes one operation
// at a time.
//
// Usage: ./p1 <opfile> <nops> <nthreads> [blocksize=1024] [static|cyclic|dynamic] [chunk=1]
//   (defaults: block size 1024, dynamic, chunk 1 = the best configuration found)
//   static   : contiguous block of nops/nthreads ops per thread
//   cyclic   : static round-robin (thread t gets ops t, t+T, t+2T, ...)
//   dynamic  : threads grab chunks of <chunk> ops from a global
//              counter with atomicAdd until the list is exhausted
//   Environment variable REPS (default 5) = number of timed repetitions.
//   Each repetition starts from an empty table.
#include "q1_common.h"

enum { SCHED_STATIC = 0, SCHED_CYCLIC = 1, SCHED_DYNAMIC = 2 };

// Insert k. Tombstones are reused: the first tombstone on the probe path is
// remembered and claimed once the scan has proven k absent (an EMPTY slot or a
// full circle). If another key takes that tombstone first, the scan restarts.
// With segregated lists no tombstone exists during inserts, so that path is idle.
__device__ int ht_insert(int *T, int n, int k)
{
	while (true) {
		int i = k % n, tomb = -1, p;                    // start at the home slot h(k)
		for (p = 0; p < n; p++, i = next_slot(i, n)) {
			int v = T[i];                                // plain read first: cheap
			if (v == k) return 0;                        // already in the set
			if (v == TOMB) { if (tomb < 0) tomb = i; continue; }   // remember the first tombstone
			if (v != EMPTY) continue;                    // another key: keep probing
			if (tomb >= 0) break;                        // k is absent: go claim the tombstone
			int old = atomicCAS(&T[i], EMPTY, k);        // "if still EMPTY, put k" -- atomically
			if (old == EMPTY) return 1;                  // we won the slot: inserted
			if (old == k) return 0;                      // another thread inserted k just now
			// another key won the slot: keep probing
		}
		if (tomb < 0) return 0;                          // full circle, no EMPTY or TOMB: table full
		int old = atomicCAS(&T[tomb], TOMB, k);          // reuse the first tombstone on the path
		if (old == TOMB) return 1;
		if (old == k) return 0;                          // another insert of k took it
		// another key took the tombstone first: rescan from the home slot
	}
}

__device__ __forceinline__ int ht_delete(int *T, int n, int k)
{
	int i = k % n;
	for (int p = 0; p < n; p++, i = next_slot(i, n)) {
		int v = T[i];
		if (v == EMPTY) return 0; // end of probe chain
		if (v == k) return atomicCAS(&T[i], k, TOMB) == k;
	}
	return 0;
}

__device__ __forceinline__ int ht_search(const int *T, int n, int k)
{
	int i = k % n;
	for (int p = 0; p < n; p++, i = next_slot(i, n)) {
		int v = T[i];
		if (v == k) return 1;
		if (v == EMPTY) return 0;
	}
	return 0;
}

__device__ __forceinline__ void do_op(int type, const int2 *ops, int idx, int *T, int n, unsigned char *res)
{
	int k = ops[idx].y;
	int r;
	if (type == OP_INSERT) r = ht_insert(T, n, k);
	else if (type == OP_DELETE) r = ht_delete(T, n, k);
	else r = ht_search(T, n, k);
	res[idx] = (unsigned char)r;
}

// One launch handles one segment [start, start+cnt) of identical operations.
__global__ void p1_kernel(int sched, int type, const int2 *ops, int start, int cnt,
			  int *T, int n, unsigned char *res, int chunk, int *counter, long long nthr_ll)
{
	long long tid_ll = (long long)blockIdx.x * blockDim.x + threadIdx.x;
	if (tid_ll >= nthr_ll) return; // last block may be partially used
	int tid = (int)tid_ll;
	int nthr = (int)min(nthr_ll, (long long)2147483647);

	if (sched == SCHED_STATIC) {
		int per = (cnt + nthr - 1) / nthr;
		long long lo = (long long)tid * per;
		long long hi = min(lo + per, (long long)cnt);
		for (long long e = lo; e < hi; e++) do_op(type, ops, start + (int)e, T, n, res);
	} else if (sched == SCHED_CYCLIC) {
		for (int e = tid; e < cnt; e += nthr) do_op(type, ops, start + e, T, n, res);
	} else {
		while (true) {
			int s = atomicAdd(counter, chunk);
			if (s >= cnt) break;
			int hi = min(s + chunk, cnt);
			for (int e = s; e < hi; e++) do_op(type, ops, start + e, T, n, res);
		}
	}
}

int main(int argc, char *argv[])
{
	if (argc < 4) {
		fprintf(stderr, "Usage: %s <opfile> <nops> <nthreads> [blocksize=1024] [static|cyclic|dynamic] [chunk=1]\n", argv[0]);
		return 1;
	}
	const char *fname = argv[1];
	int nops = atoi(argv[2]);
	long long nthreads = atoll(argv[3]);
	int bs = (argc > 4) ? atoi(argv[4]) : 1024;
	int sched = SCHED_DYNAMIC;
	if (argc > 5) {
		if (!strcmp(argv[5], "static")) sched = SCHED_STATIC;
		else if (!strcmp(argv[5], "cyclic")) sched = SCHED_CYCLIC;
		else if (!strcmp(argv[5], "dynamic")) sched = SCHED_DYNAMIC;
		else { fprintf(stderr, "Unknown schedule %s\n", argv[5]); return 1; }
	}
	int chunk = (argc > 6) ? atoi(argv[6]) : 1;
	int reps = env_int("REPS", 5);
	const char *sched_name[3] = { "static", "cyclic", "dynamic" };

	if (nops <= 0 || nthreads <= 0 || bs <= 0 || chunk <= 0) { fprintf(stderr, "Arguments must be positive\n"); return 1; }
	if (bs > 1024) { fprintf(stderr, "Block size must be <= 1024\n"); return 1; }
	if (nthreads < bs) bs = (int)nthreads;
	long long nblocks = (nthreads + bs - 1) / bs; // any thread count works
	if (nblocks > 2147483647LL) { fprintf(stderr, "Too many blocks\n"); return 1; }

	Workload w;
	load_workload(fname, nops, w);
	VerifyPlan vp;
	build_verify_plan(w, vp);          // expected results, computed once (not timed)
	DeviceState d;
	device_setup(w, d);

	cudaEvent_t ev0, ev1;
	CUDA_CHECK(cudaEventCreate(&ev0));
	CUDA_CHECK(cudaEventCreate(&ev1));

	std::vector<int> zeros(w.segs.size(), 0);
	double total_ms = 0, best_ms = 1e30;
	bool ok = true;
	for (int r = 0; r < reps; r++) {
		// Not timed: empty table, reset dynamic counters
		CUDA_CHECK(cudaMemset(d.table, 0xFF, sizeof(int) * (size_t)w.n)); // every int = -1 = EMPTY
		CUDA_CHECK(cudaMemcpy(d.counters, zeros.data(), sizeof(int) * zeros.size(), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaEventRecord(ev0));
		for (size_t s = 0; s < w.segs.size(); s++)
			p1_kernel<<<(int)nblocks, bs>>>(sched, w.segs[s].type, d.ops, w.segs[s].start, w.segs[s].count,
							d.table, w.n, d.res, chunk, d.counters + s, nthreads);
		CUDA_CHECK(cudaEventRecord(ev1));
		CUDA_CHECK(cudaEventSynchronize(ev1));
		CUDA_CHECK(cudaGetLastError());

		float ms;
		CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
		total_ms += ms;
		if (ms < best_ms) best_ms = ms;
		ok = verify(w, vp, d.res, d.table) && ok;   // every repetition, outside the timer
	}
	double avg_ms = total_ms / reps;

	printf("P1 sched=%s chunk=%d threads=%lld blocks=%lld blocksize=%d nops=%d (ins=%d del=%d srch=%d) n=%d "
	       "time=%.3f ms (min %.3f, %d reps) throughput=%.2f Mops/s verify=%s\n",
	       sched_name[sched], sched == SCHED_DYNAMIC ? chunk : 0, nthreads, nblocks, bs, nops,
	       w.count[0], w.count[1], w.count[2], w.n, avg_ms, best_ms, reps, nops / (avg_ms * 1e3),
	       ok ? "OK" : "FAILED");
	printf("CSV,P1,%s,%d,%lld,%lld,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%s\n",
	       sched_name[sched], sched == SCHED_DYNAMIC ? chunk : 0, nthreads, nblocks, bs, nops,
	       w.count[0], w.count[1], w.count[2], w.n, avg_ms, best_ms, ok ? "OK" : "FAILED");

	device_free(d);
	return ok ? 0 : 2;
}
