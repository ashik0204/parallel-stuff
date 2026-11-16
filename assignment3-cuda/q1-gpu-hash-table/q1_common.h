// q1_common.h -- host-side code and device hash-table primitives shared
// by P1 (one thread per operation) and P2 (one thread group per operation).
//
// Hash table: open addressing with linear probing, array of n ints,
// h(k) = k mod n.  Keys are non-negative ints, so two negative values are
// used as markers:
//     EMPTY = -1  slot never used (all bytes 0xFF, so cudaMemset can set it)
//     TOMB  = -2  slot whose key was deleted (tombstone)
//
// Within one kernel a slot only moves forward: EMPTY/TOMB -> key in an insert
// kernel, key -> TOMB in a delete kernel.  This is what makes the lock-free
// algorithms correct:
//   - insert claims an EMPTY slot with atomicCAS(EMPTY -> k), or, once it has
//     seen that k is absent, reuses the first tombstone on its path with
//     atomicCAS(TOMB -> k); a failed CAS tells us who won the slot (if it was
//     another insert of k, k is present);
//   - all inserts of k walk the same probe sequence and compete for the same
//     first free slot, so a key can never get two slots;
//   - delete leaves a tombstone instead of EMPTY, so probe chains of other
//     keys are never broken;
//   - a stale read can only show an older value of a slot, and the CAS that
//     follows returns the real value.
//
// The operation list is segregated (all inserts, then deletes, then
// searches).  The host splits it into runs of identical operations and
// launches one kernel per run.  The kernel boundary acts as a global
// barrier, so inserts and deletes never overlap in time.
#ifndef Q1_COMMON_H
#define Q1_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cuda_runtime.h>

#define OP_INSERT 0
#define OP_DELETE 1
#define OP_SEARCH 2

#define EMPTY (-1)
#define TOMB  (-2)

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
	fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
	exit(1); } } while (0)

__attribute__((unused)) static const char *op_name[3] = { "insert", "delete", "search" };

struct Segment { int type; int start; int count; };

struct Workload {
	int nops;
	int n; // table size = #distinct insert keys
	std::vector<int2> ops; // .x = op, .y = key
	std::vector<Segment> segs;
	int count[3];
};

// Read the first nops operations from the binary file produced by gen_ops.
static void load_workload(const char *fname, int nops, Workload &w)
{
	FILE *fp = fopen(fname, "rb");
	if (!fp) { perror(fname); exit(1); }
	w.nops = nops;
	w.ops.resize(nops);
	size_t got = fread(w.ops.data(), sizeof(int2), nops, fp);
	fclose(fp);
	if ((int)got != nops) {
		fprintf(stderr, "File %s has only %zu operations, %d requested\n", fname, got, nops);
		exit(1);
	}

	// Split into maximal runs of the same operation type
	w.count[0] = w.count[1] = w.count[2] = 0;
	std::vector<int> keys;
	for (int i = 0; i < nops; i++) {
		int t = w.ops[i].x;
		if (t < 0 || t > 2 || w.ops[i].y < 0) {
			fprintf(stderr, "Bad record %d: op=%d key=%d\n", i, t, w.ops[i].y);
			exit(1);
		}
		w.count[t]++;
		if (t == OP_INSERT) keys.push_back(w.ops[i].y);
		if (w.segs.empty() || w.segs.back().type != t) w.segs.push_back(Segment{t, i, 0});
		w.segs.back().count++;
	}

	// A segregated list has at most 3 segments; a mixed list is still handled
	// correctly (one kernel per segment), only more slowly.
	if (w.segs.size() > 3)
		fprintf(stderr, "warning: list is not segregated (%zu segments); "
		        "running one kernel per segment\n", w.segs.size());

	// Table size = number of distinct keys inserted
	std::sort(keys.begin(), keys.end());
	w.n = (int)(std::unique(keys.begin(), keys.end()) - keys.begin());
	if (w.n == 0) { fprintf(stderr, "No inserts in workload\n"); exit(1); }
}

// ---------------- device: probe helpers ----------------

__device__ __forceinline__ int next_slot(int i, int n) { return (i + 1 == n) ? 0 : i + 1; }

// Device buffers and results of a run
struct DeviceState {
	int2 *ops;
	int *table;
	unsigned char *res;
	int *counters; // one per segment, used by dynamic scheduling
};

static void device_setup(const Workload &w, DeviceState &d)
{
	size_t need = sizeof(int2) * (size_t)w.nops + sizeof(int) * (size_t)w.n + w.nops;
	size_t freeb, totalb;
	CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
	if (need > freeb) {
		fprintf(stderr, "Need %zu MB on GPU, only %zu MB free\n", need >> 20, freeb >> 20);
		exit(1);
	}
	CUDA_CHECK(cudaMalloc(&d.ops, sizeof(int2) * (size_t)w.nops));
	CUDA_CHECK(cudaMalloc(&d.table, sizeof(int) * (size_t)w.n));
	CUDA_CHECK(cudaMalloc(&d.res, w.nops));
	CUDA_CHECK(cudaMalloc(&d.counters, sizeof(int) * w.segs.size()));
	CUDA_CHECK(cudaMemcpy(d.ops, w.ops.data(), sizeof(int2) * (size_t)w.nops, cudaMemcpyHostToDevice));
}

static void device_free(DeviceState &d)
{
	cudaFree(d.ops); cudaFree(d.table); cudaFree(d.res); cudaFree(d.counters);
}

// ---------------------------------------------------------------- verification
// Exact expected results, computed once on the CPU by replaying the list
// segment by segment (sequential semantics between segments). Inside one
// segment the GPU runs all ops concurrently, so per key in a segment:
//   insert: key present before -> all its inserts return 0, absent -> exactly one returns 1
//   delete: key present before -> exactly one returns 1,    absent -> all return 0
//   search: every search returns (key present before)
// Only WHICH duplicate insert/delete wins is left open (that depends on timing).
struct VerifyPlan {
	std::vector<int> perm;        // op indices, grouped by (segment, key)
	std::vector<int> gstart;      // group g = perm[gstart[g] .. gstart[g+1])
	std::vector<int> gexp;        // expected number of 1s in group g (-1 = all of them)
	std::vector<int> final_keys;  // sorted keys that must be in the table at the end
};

static void build_verify_plan(const Workload &w, VerifyPlan &vp)
{
	std::vector<int> S;           // sorted set of keys present before the current segment
	vp.perm.clear(); vp.gstart.clear(); vp.gexp.clear();
	vp.perm.reserve(w.nops);
	for (const Segment &sg : w.segs) {
		size_t base = vp.perm.size();
		for (int i = sg.start; i < sg.start + sg.count; i++) vp.perm.push_back(i);
		std::sort(vp.perm.begin() + base, vp.perm.end(),
		          [&](int a, int b) { return w.ops[a].y < w.ops[b].y; });
		std::vector<int> seg_keys;                      // distinct keys of this segment, sorted
		for (size_t j = base; j < vp.perm.size(); j++) {
			int k = w.ops[vp.perm[j]].y;
			if (j > base && k == w.ops[vp.perm[j - 1]].y) continue;   // same key, same group
			bool present = std::binary_search(S.begin(), S.end(), k);
			vp.gstart.push_back((int)j);
			if (sg.type == OP_INSERT) vp.gexp.push_back(present ? 0 : 1);
			else if (sg.type == OP_DELETE) vp.gexp.push_back(present ? 1 : 0);
			else vp.gexp.push_back(present ? -1 : 0);
			seg_keys.push_back(k);
		}
		std::vector<int> next;                          // set after this segment
		if (sg.type == OP_INSERT) std::set_union(S.begin(), S.end(), seg_keys.begin(), seg_keys.end(), std::back_inserter(next));
		else if (sg.type == OP_DELETE) std::set_difference(S.begin(), S.end(), seg_keys.begin(), seg_keys.end(), std::back_inserter(next));
		else continue;                                  // searches do not change the set
		S.swap(next);
	}
	vp.gstart.push_back((int)vp.perm.size());           // sentinel: end of the last group
	vp.final_keys.swap(S);
}

// Check the GPU's results and table against the plan. Called after EVERY timed run.
static bool verify(const Workload &w, const VerifyPlan &vp, const unsigned char *d_res, const int *d_table)
{
	std::vector<unsigned char> res(w.nops);
	std::vector<int> T(w.n);
	CUDA_CHECK(cudaMemcpy(res.data(), d_res, w.nops, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(T.data(), d_table, sizeof(int) * (size_t)w.n, cudaMemcpyDeviceToHost));
	bool ok = true;

	// 1. Every op's return value (per key group)
	long long bad = 0;
	for (size_t g = 0; g + 1 < vp.gstart.size(); g++) {
		int sum = 0, size = vp.gstart[g + 1] - vp.gstart[g];
		for (int j = vp.gstart[g]; j < vp.gstart[g + 1]; j++) sum += res[vp.perm[j]];
		int expect = vp.gexp[g] < 0 ? size : vp.gexp[g];
		if (sum != expect && bad++ < 3) {
			int op = vp.perm[vp.gstart[g]];
			fprintf(stderr, "verify: %s of key %d (op %d): %d of %d returned 1, expected %d\n",
			        op_name[w.ops[op].x], w.ops[op].y, op, sum, size, expect);
		}
	}
	if (bad) { fprintf(stderr, "verify: %lld key groups with wrong return values\n", bad); ok = false; }

	// 2. Table holds exactly the expected keys: none missing, none extra, none twice
	std::vector<int> live;
	for (int v : T) if (v >= 0) live.push_back(v);
	std::sort(live.begin(), live.end());
	if (live != vp.final_keys) {
		fprintf(stderr, "verify: table holds %zu keys, expected %zu (or wrong/duplicate keys)\n",
		        live.size(), vp.final_keys.size());
		ok = false;
	}

	// 3. Reachability: no EMPTY between a key's home and its slot (else a search stops early).
	//    dist = steps back to the nearest EMPTY; the 2nd pass handles wrap-around.
	long long n = w.n, dist = n;
	for (int pass = 0; pass < 2; pass++)
		for (long long i = 0; i < n; i++) {
			dist = (T[i] == EMPTY) ? 0 : std::min(dist + 1, n);
			long long disp = (i - T[i] % n + n) % n;          // how far the key is from home
			if (pass == 1 && T[i] >= 0 && dist < n && disp >= dist) {
				fprintf(stderr, "verify: key %d at slot %lld is unreachable\n", T[i], i); ok = false; break;
			}
		}
	return ok;
}

static int env_int(const char *name, int def)
{
	const char *s = getenv(name);
	return s ? atoi(s) : def;
}

__attribute__((unused)) static bool is_pow2(long long x) { return x > 0 && (x & (x - 1)) == 0; }

#endif
