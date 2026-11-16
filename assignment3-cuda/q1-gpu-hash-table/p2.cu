// p2.cu -- Question 1, Program P2.
// Concurrent open-addressed hash table: each operation is handled by a
// group of g threads (g = 1, 2, 4, 8, 16, 32; groups never span warps).
//
// The group probes a window of g consecutive slots at once: lane l reads
// T[s + l].  Consecutive lanes read consecutive addresses, so one probe
// step of the whole group is one coalesced memory access.  The group first
// asks with one __ballot_sync whether any lane saw k, EMPTY or TOMB, and only
// then looks closer with further ballots:
//   - some lane saw k        -> found (search/insert-duplicate/delete target)
//   - some lane saw EMPTY    -> end of chain (search/delete fail; insert reuses
//                               the first tombstone seen on its path, or else
//                               tries to CAS the empty slots in window order)
//   - some lane saw TOMB     -> remembered by an insert as a reusable slot
//   - otherwise              -> advance the window by g
// Only one lane (the leader chosen by __ffs) executes an atomicCAS; the
// outcome is broadcast to the group with __shfl_sync.
//
// Usage: ./p2 <opfile> <nops> <nthreads> [blocksize=128] [groupsize=32] [static|dynamic] [chunk=1]
//   static  : group q does operations q, q+Q, q+2Q, ... (Q = number of groups), the default
//   dynamic : the group's first lane takes <chunk> operations at a time from a global
//             counter with atomicAdd (best: dynamic, chunk 1)
//   Environment variable REPS (default 5) = number of timed repetitions.
#include "q1_common.h"

// Lane bookkeeping for a group: lane = lane in warp, glane = lane in group,
// gmask = bit mask of the group's lanes inside the warp.
struct Grp {
	int lane, glane;
	unsigned gmask;
	int g;
};

__device__ __forceinline__ int wrap(int p, int n) { while (p >= n) p -= n; return p; }

// Each probe step reads a window of g consecutive slots (one coalesced load) and
// first asks with ONE ballot whether any lane saw something that matters (k, EMPTY
// or TOMB). In a full table most windows hold only other keys, so the detailed
// ballots below run only for the few windows where something happens.

// Insert k. Tombstones are reused: the first tombstone on the probe path is
// remembered and claimed once the scan has proven k absent (an EMPTY slot or a
// full circle). If another key takes that tombstone first, the scan restarts.
// With segregated lists no tombstone exists during inserts, so that path is idle.
__device__ __forceinline__ int grp_insert(int *T, int n, int k, const Grp &G)
{
	while (true) {
		int s = k % n, tomb = -1;
		bool claim_tomb = false;
		for (int probed = 0; probed < n; probed += G.g) {
			int pos = wrap(s + G.glane, n);
			int v = T[pos];
			if (__ballot_sync(G.gmask, v == k || v == EMPTY || v == TOMB) & G.gmask) {
				if (__ballot_sync(G.gmask, v == k) & G.gmask) return 0; // already present
				unsigned emp = __ballot_sync(G.gmask, v == EMPTY) & G.gmask;
				unsigned tb_all = 0;
				if (tomb < 0) {
					tb_all = __ballot_sync(G.gmask, v == TOMB) & G.gmask;
					unsigned tb = emp ? tb_all & ((emp & (0u - emp)) - 1) : tb_all; // before the first EMPTY
					if (tb) tomb = __shfl_sync(G.gmask, pos, __ffs(tb) - 1);
				}
				if (emp && tomb >= 0) { claim_tomb = true; break; } // k absent: reuse the tombstone
				// Try the empty slots of the window in probe order. Slots before the
				// first empty one hold other keys, so the first EMPTY is where k belongs.
				while (emp) {
					int leader = __ffs(emp) - 1;
					int old = 0;
					if (G.lane == leader) old = atomicCAS(&T[pos], EMPTY, k);
					old = __shfl_sync(G.gmask, old, leader);
					if (old == EMPTY) return 1; // inserted
					if (old == k) return 0; // lost race to another insert of k
					emp &= emp - 1; // slot went to another key
				}
				// Every EMPTY of the window went to other keys, so tombstones after them
				// are now on the path too (P1 does the same by continuing its scan).
				if (tomb < 0 && tb_all) tomb = __shfl_sync(G.gmask, pos, __ffs(tb_all) - 1);
			}
			s = wrap(s + G.g, n);
		}
		if (!claim_tomb && tomb < 0) return 0; // full circle, no EMPTY, no TOMB: table full
		int leader = __ffs(G.gmask) - 1, old = 0; // full circle with a tombstone, or EMPTY seen
		if (G.lane == leader) old = atomicCAS(&T[tomb], TOMB, k);
		old = __shfl_sync(G.gmask, old, leader);
		if (old == TOMB) return 1; // reused the tombstone
		if (old == k) return 0; // another insert of k took it
		// another key took the tombstone first: rescan from the home slot
	}
}

__device__ __forceinline__ int grp_delete(int *T, int n, int k, const Grp &G)
{
	int s = k % n;
	for (int probed = 0; probed < n; probed += G.g) {
		int pos = wrap(s + G.glane, n);
		int v = T[pos];
		if (__ballot_sync(G.gmask, v == k || v == EMPTY) & G.gmask) {
			unsigned hit = __ballot_sync(G.gmask, v == k) & G.gmask;
			if (!hit) return 0; // an EMPTY ends the chain: not present
			int leader = __ffs(hit) - 1;
			int old = 0;
			if (G.lane == leader) old = atomicCAS(&T[pos], k, TOMB);
			old = __shfl_sync(G.gmask, old, leader);
			return old == k;
		}
		s = wrap(s + G.g, n);
	}
	return 0;
}

__device__ __forceinline__ int grp_search(const int *T, int n, int k, const Grp &G)
{
	int s = k % n;
	for (int probed = 0; probed < n; probed += G.g) {
		int pos = wrap(s + G.glane, n);
		int v = T[pos];
		if (__ballot_sync(G.gmask, v == k || v == EMPTY) & G.gmask)
			return (__ballot_sync(G.gmask, v == k) & G.gmask) != 0; // k before the chain ends?
		s = wrap(s + G.g, n);
	}
	return 0;
}

__device__ __forceinline__ void grp_op(int type, const int2 *ops, int idx, int *T, int n,
                                        unsigned char *res, const Grp &G)
{
	int k = ops[idx].y; // same address for the group: one broadcast load
	int r;
	if (type == OP_INSERT) r = grp_insert(T, n, k, G);
	else if (type == OP_DELETE) r = grp_delete(T, n, k, G);
	else r = grp_search(T, n, k, G);
	if (G.glane == 0) res[idx] = (unsigned char)r;
}

// static : groups take operations cyclically (group q: ops q, q+Q, ...)
// dynamic: the group's first lane grabs `chunk` ops from a shared counter with
//          atomicAdd and broadcasts the start to its group with __shfl_sync
__global__ void p2_kernel(int dynamic, int type, const int2 *ops, int start, int cnt, int *T, int n,
			  unsigned char *res, int g, int ngroups, int chunk, int *counter)
{
	Grp G;
	G.g = g;
	G.lane = threadIdx.x & 31;
	G.glane = G.lane & (g - 1);
	int gfirst = G.lane & ~(g - 1);
	G.gmask = (g == 32) ? 0xffffffffu : (((1u << g) - 1) << gfirst);

	long long group_ll = ((long long)blockIdx.x * blockDim.x + threadIdx.x) / g;
	if (group_ll >= ngroups) return; // whole group leaves together
	int group = (int)group_ll;

	// All lanes of a group handle the same operations in the same order, so they
	// stay together at every ballot/shuffle.
	if (!dynamic) {
		for (int e = group; e < cnt; e += ngroups) grp_op(type, ops, start + e, T, n, res, G);
		return;
	}
	while (true) {
		int first = 0;
		if (G.lane == gfirst) first = atomicAdd(counter, chunk); // one atomic per group
		first = __shfl_sync(G.gmask, first, gfirst);
		if (first >= cnt) break;
		int last = first + chunk < cnt ? first + chunk : cnt;
		for (int e = first; e < last; e++) grp_op(type, ops, start + e, T, n, res, G);
	}
}

int main(int argc, char *argv[])
{
	if (argc < 4) {
		fprintf(stderr, "Usage: %s <opfile> <nops> <nthreads> [blocksize=128] [groupsize=32] [static|dynamic] [chunk=1]\n", argv[0]);
		return 1;
	}
	const char *fname = argv[1];
	int nops = atoi(argv[2]);
	long long nthreads = atoll(argv[3]);
	int bs = (argc > 4) ? atoi(argv[4]) : 128;
	int g = (argc > 5) ? atoi(argv[5]) : 32;
	int dynamic = 0; // static (cyclic) is the default
	if (argc > 6) {
		if (!strcmp(argv[6], "dynamic")) dynamic = 1;
		else if (strcmp(argv[6], "static")) { fprintf(stderr, "Schedule must be static or dynamic\n"); return 1; }
	}
	int chunk = (argc > 7) ? atoi(argv[7]) : 1;
	int reps = env_int("REPS", 5);

	if (nops <= 0 || nthreads <= 0 || bs <= 0 || chunk <= 0) { fprintf(stderr, "Arguments must be positive\n"); return 1; }
	if (!is_pow2(g) || g > 32) { fprintf(stderr, "Group size must be a power of two <= 32\n"); return 1; }
	if (bs > 1024) { fprintf(stderr, "Block size must be <= 1024\n"); return 1; }
	// Any thread count / block size is accepted: the group size is capped
	// by the thread count, threads that do not form a full group stay idle,
	// and the block size is rounded up to a multiple of the group size so a
	// group never straddles two warps/blocks.
	while (g > nthreads) g >>= 1;
	long long ngroups = nthreads / g;
	if (ngroups > 2147483647LL) { fprintf(stderr, "Too many groups\n"); return 1; }
	if (nthreads < bs) bs = (int)nthreads;
	if (bs % g) bs = ((bs + g - 1) / g) * g;
	if (bs > 1024) bs = 1024;
	long long nblocks = (ngroups * g + bs - 1) / bs;
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

	double total_ms = 0, best_ms = 1e30;
	bool ok = true;
	for (int r = 0; r < reps; r++) {
		CUDA_CHECK(cudaMemset(d.table, 0xFF, sizeof(int) * (size_t)w.n)); // EMPTY
		CUDA_CHECK(cudaMemset(d.counters, 0, sizeof(int) * w.segs.size())); // dynamic counters
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaEventRecord(ev0));
		for (size_t s = 0; s < w.segs.size(); s++)
			p2_kernel<<<(int)nblocks, bs>>>(dynamic, w.segs[s].type, d.ops, w.segs[s].start, w.segs[s].count,
							d.table, w.n, d.res, g, (int)ngroups, chunk, d.counters + s);
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

	char sched[32];
	if (dynamic) snprintf(sched, sizeof sched, "dynamic-%d", chunk); else snprintf(sched, sizeof sched, "group");
	printf("P2 %s group=%d threads=%lld blocks=%lld blocksize=%d nops=%d (ins=%d del=%d srch=%d) n=%d "
	       "time=%.3f ms (min %.3f, %d reps) throughput=%.2f Mops/s verify=%s\n",
	       sched, g, nthreads, nblocks, bs, nops, w.count[0], w.count[1], w.count[2], w.n,
	       avg_ms, best_ms, reps, nops / (avg_ms * 1e3), ok ? "OK" : "FAILED");
	printf("CSV,P2,%s,%d,%lld,%lld,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%s\n",
	       sched, g, nthreads, nblocks, bs, nops, w.count[0], w.count[1], w.count[2], w.n,
	       avg_ms, best_ms, ok ? "OK" : "FAILED");

	device_free(d);
	return ok ? 0 : 2;
}
