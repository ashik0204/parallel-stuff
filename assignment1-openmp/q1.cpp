// CS433 Assignment #1, Question 1 (OpenMP)
// Usage: ./q1 <input file> <output file> <number of threads>
//
// Rules
// - (i)   end points of an edge: different colors
// - (ii)  edge vs its end points: different colors
// - (iii) the 3 edges of a triangle: all different
//
// Idea: conflict graph H
// - nodes = n vertices + m edges of G
// - vertex u -- vertex v : (u, v) is an edge
// - vertex u -- edge e   : u is an end point of e
// - edge e   -- edge f   : e, f in a common triangle
// - proper coloring of H = valid coloring of G
//
// Steps
// 1. build H in parallel (triangles by merging sorted lists)
// 2. batched parallel DSATUR (independent set of max-saturation nodes per round)
// 3. parallel top-class elimination (local search)
// 4. clique lower bound; exact branch and bound for small graphs
// - deterministic: same output for any thread count
//
// Terms
// - CSR = compressed sparse row: all neighbor lists packed into one array

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <atomic>
#include <memory>

typedef long long ll;

static int N, M, V;                     // vertices, edges, nodes of H
static std::vector<int> eu, ev;         // edge end points, eu < ev
static std::vector<ll> gOff;            // CSR of G (sorted lists)
static std::vector<int> gAdj, gEid;     // neighbor, connecting edge id
static std::vector<ll> hOff;            // CSR of H
static std::vector<int> hAdj;
static std::vector<int> color;          // color per node of H
static int nthreads;

static const int EXACT_MAX_NODES = 400;   // exact search if |V(H)| <= this
static const ll EXACT_BUDGET = 20000000;  // max search tree nodes
static const int MAX_BLOCKERS = 4;        // elimination: max neighbors moved
static const size_t SMALL_ROUND = 128;    // smaller DSATUR rounds: one thread
static const double ORIENT_MIN_AVG_DEGREE = 16;  // dense: each triangle found once

static inline int HDeg(int v) { return (int)(hOff[v + 1] - hOff[v]); }
static int maxDegH = 0;                 // max degree in H

// Timer
// - measures computation only
// - paused around allocation / array initialization (assignment rule)
static double measured = 0.0, runningSince = 0.0;
static inline void ClockStart() { runningSince = omp_get_wtime(); }
static inline void ClockStop() { measured += omp_get_wtime() - runningSince; }

// ---------------------------------------------------------------------------
// Input
// - line 1: n
// - line k+2: neighbors i > k of vertex k
// ---------------------------------------------------------------------------

static void ReadInput(const char *fname) {
    FILE *f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", fname); exit(1); }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> buf(size + 1);
    size_t got = fread(buf.data(), 1, size, f);
    buf[got] = '\0';
    fclose(f);

    const char *s = buf.data(), *end = s + got;
    while (s < end && (*s < '0' || *s > '9')) s++;
    N = 0;
    while (s < end && *s >= '0' && *s <= '9') N = N * 10 + (*s++ - '0');
    while (s < end && *s != '\n') s++;  // skip rest of line 1

    std::vector<uint64_t> keys;
    int k = -1;  // current vertex (line)
    while (s < end) {
        if (*s == '\n') { k++; s++; continue; }
        if (*s >= '0' && *s <= '9') {
            ll j = 0;
            while (s < end && *s >= '0' && *s <= '9') {
                if (j <= N) j = j * 10 + (*s - '0');  // cap: no overflow
                s++;
            }
            if (k >= 0 && k < N && j < N && j != k) {
                int a = std::min<ll>(k, j), b = std::max<ll>(k, j);
                keys.push_back((uint64_t)a << 32 | (uint32_t)b);
            }
            continue;
        }
        s++;  // anything else = separator
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    M = (int)keys.size();
    eu.resize(M);
    ev.resize(M);
    for (int e = 0; e < M; e++) { eu[e] = (int)(keys[e] >> 32); ev[e] = (int)(keys[e] & 0xffffffffu); }

    // CSR of G (edges sorted -> lists sorted)
    gOff.assign(N + 1, 0);
    for (int e = 0; e < M; e++) { gOff[eu[e] + 1]++; gOff[ev[e] + 1]++; }
    for (int v = 0; v < N; v++) gOff[v + 1] += gOff[v];
    gAdj.resize(2 * (size_t)M);
    gEid.resize(2 * (size_t)M);
    std::vector<ll> pos(gOff.begin(), gOff.end() - 1);
    for (int e = 0; e < M; e++) {
        gAdj[pos[eu[e]]] = ev[e]; gEid[pos[eu[e]]++] = e;
        gAdj[pos[ev[e]]] = eu[e]; gEid[pos[ev[e]]++] = e;
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

// Parallel exclusive prefix sum
// - off[i] = cnt[0] + ... + cnt[i-1]
// - off, part (nthreads + 1) allocated by caller
static void PrefixSum(const std::vector<ll> &cnt, std::vector<ll> &off, std::vector<ll> &part) {
    int n = (int)cnt.size();
    #pragma omp parallel num_threads(nthreads)
    {
        int t = omp_get_thread_num(), T = omp_get_num_threads(); // T may be < nthreads
        int lo = (int)((ll)n * t / T), hi = (int)((ll)n * (t + 1) / T);
        ll s = 0;
        for (int i = lo; i < hi; i++) s += cnt[i];
        part[t + 1] = s;
        #pragma omp barrier
        #pragma omp single
        for (int i = 0; i < T; i++) part[i + 1] += part[i];
        s = part[t];
        for (int i = lo; i < hi; i++) { off[i] = s; s += cnt[i]; }
        if (t == T - 1) off[n] = s;
    }
}

// Smallest color not used by v's colored neighbors
// - mark / stamp: per-thread scratch
static inline int FirstFit(int v, std::vector<int> &mark, int &stamp) {
    ++stamp;
    int d = HDeg(v);
    for (ll i = hOff[v]; i < hOff[v + 1]; i++) {
        int c = color[hAdj[i]];
        if (c >= 0 && c <= d) mark[c] = stamp;
    }
    int c = 0;
    while (mark[c] == stamp) c++;
    return c;
}

// Per-thread list, padded
// - push_back writes the vector header
// - padding: headers of different threads never share a cache line
// - avoids false sharing
struct PaddedList {
    std::vector<int> v;
    char pad[64];
};

// Hash: tie-break between equal degrees
static inline uint32_t Hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}

// ---------------------------------------------------------------------------
// Merge-intersection of sorted ranges A[i..ie), A[j..je)
// - f(p, q) for every common value, in increasing order
// ---------------------------------------------------------------------------

template <class F>
static inline void Intersect(const int *A, ll i, ll ie, ll j, ll je, F &&f) {
    while (i < ie && j < je) {
        if (A[i] < A[j]) i++;
        else if (A[i] > A[j]) j++;
        else { f(i, j); i++; j++; }
    }
}

// ---------------------------------------------------------------------------
// Step 1: build conflict graph H
// - count pass (list sizes) -> prefix sum (offsets) -> fill pass
// Sparse mode
// - each edge (u, v) merges lists of u and v
// - every triangle found 3 times, no atomics
// Dense mode (avg degree >= ORIENT_MIN_AVG_DEGREE)
// - rank vertices by (degree, id), keep higher-ranked neighbors only
// - each triangle found once, from its lowest-ranked vertex
// - entries added to all 3 edges via atomic cursors
// - edge lists then sorted by third vertex: H identical to sparse mode
// - measured: ~2x faster on dense graphs, slower on sparse ones
// ---------------------------------------------------------------------------

static ll BuildConflictGraph() {
    ClockStop();  // allocation: not timed
    V = N + M;
    const bool oriented = N > 0 && 2.0 * M / N >= ORIENT_MIN_AVG_DEGREE;
    std::vector<ll> cnt(V);
    std::vector<ll> part(nthreads + 1, 0), fcnt, fOff;
    std::vector<int> fAdj, fEid, tri;
    hOff.assign(V + 1, 0);
    if (oriented) { fcnt.assign(N, 0); fOff.assign(N + 1, 0); tri.assign(M, 0); }
    ClockStart();
    ll triangles = 0;
    const int *A = gAdj.data();

    if (!oriented) {
        // degree in H: vertex = 2 * deg, edge = 2 + 2 * #triangles
        ll triSum = 0;
        #pragma omp parallel for schedule(dynamic, 1024) num_threads(nthreads) reduction(+ : triSum)
        for (int x = 0; x < V; x++) {
            if (x < N) { cnt[x] = 2 * (gOff[x + 1] - gOff[x]); continue; }
            int e = x - N, u = eu[e], v = ev[e];
            ll t = 0;
            Intersect(A, gOff[u], gOff[u + 1], gOff[v], gOff[v + 1], [&](ll, ll) { t++; });
            cnt[x] = 2 + 2 * t;
            triSum += t;
        }
        triangles = triSum / 3;  // each triangle seen from its 3 edges
    } else {
        auto deg = [&](int v) { return gOff[v + 1] - gOff[v]; };
        auto above = [&](int a, int b) { return deg(a) > deg(b) || (deg(a) == deg(b) && a > b); };
        #pragma omp parallel for schedule(dynamic, 1024) num_threads(nthreads)
        for (int u = 0; u < N; u++) {
            ll c = 0;
            for (ll p = gOff[u]; p < gOff[u + 1]; p++) c += above(gAdj[p], u);
            fcnt[u] = c;
        }
        PrefixSum(fcnt, fOff, part);
        ClockStop();  // allocation: not timed
        fAdj.assign(fOff[N], 0);
        fEid.assign(fOff[N], 0);
        ClockStart();
        #pragma omp parallel for schedule(dynamic, 1024) num_threads(nthreads)
        for (int u = 0; u < N; u++) {
            ll o = fOff[u];
            for (ll p = gOff[u]; p < gOff[u + 1]; p++)
                if (above(gAdj[p], u)) { fAdj[o] = gAdj[p]; fEid[o++] = gEid[p]; }
        }
        const int *F = fAdj.data();
        ll triSum = 0;
        #pragma omp parallel for schedule(dynamic, 64) num_threads(nthreads) reduction(+ : triSum)
        for (int u = 0; u < N; u++) {
            ll local = 0;
            for (ll p = fOff[u]; p < fOff[u + 1]; p++) {
                int v = fAdj[p], euv = fEid[p];
                Intersect(F, fOff[u], fOff[u + 1], fOff[v], fOff[v + 1], [&](ll i, ll j) {
                    int euw = fEid[i], evw = fEid[j];
                    #pragma omp atomic
                    tri[euv]++;
                    #pragma omp atomic
                    tri[euw]++;
                    #pragma omp atomic
                    tri[evw]++;
                    local++;
                });
            }
            triSum += local;
        }
        triangles = triSum;
        #pragma omp parallel for schedule(static) num_threads(nthreads)
        for (int x = 0; x < V; x++)
            cnt[x] = x < N ? 2 * deg(x) : 2 + 2 * (ll)tri[x - N];
    }

    PrefixSum(cnt, hOff, part);
    ClockStop();  // allocation: not timed
    hAdj.assign(hOff[V], 0);
    std::vector<ll> cursor;
    if (oriented) cursor.assign(M, 0);
    ClockStart();
    int md = 0;
    #pragma omp parallel for schedule(dynamic, 1024) num_threads(nthreads) reduction(max : md)
    for (int x = 0; x < V; x++) {
        ll o = hOff[x];
        md = std::max(md, (int)cnt[x]);
        if (x < N) {
            for (ll i = gOff[x]; i < gOff[x + 1]; i++) hAdj[o++] = gAdj[i];
            for (ll i = gOff[x]; i < gOff[x + 1]; i++) hAdj[o++] = N + gEid[i];
            continue;
        }
        int e = x - N, u = eu[e], v = ev[e];
        hAdj[o++] = u;
        hAdj[o++] = v;
        if (oriented) { cursor[e] = o; continue; }
        Intersect(A, gOff[u], gOff[u + 1], gOff[v], gOff[v + 1], [&](ll i, ll j) {
            hAdj[o++] = N + gEid[i];  // edge (u, w)
            hAdj[o++] = N + gEid[j];  // edge (v, w)
        });
    }
    if (oriented) {
        // add partners (f, g) to edge e's list
        // - edge containing eu[e] first (same order as sparse mode)
        auto put = [&](int e, int f, int g) {
            int a = eu[e];
            if (eu[f] != a && ev[f] != a) std::swap(f, g);
            ll p;
            #pragma omp atomic capture
            { p = cursor[e]; cursor[e] += 2; }
            hAdj[p] = N + f;
            hAdj[p + 1] = N + g;
        };
        const int *F = fAdj.data();
        #pragma omp parallel for schedule(dynamic, 64) num_threads(nthreads)
        for (int u = 0; u < N; u++) {
            for (ll p = fOff[u]; p < fOff[u + 1]; p++) {
                int v = fAdj[p], euv = fEid[p];
                Intersect(F, fOff[u], fOff[u + 1], fOff[v], fOff[v + 1], [&](ll i, ll j) {
                    int euw = fEid[i], evw = fEid[j];
                    put(euv, euw, evw);
                    put(euw, euv, evw);
                    put(evw, euv, euw);
                });
            }
        }
        // canonical order: sort each edge's pairs by third vertex
        #pragma omp parallel num_threads(nthreads)
        {
            std::vector<std::pair<int, std::pair<int, int>>> tmp;
            #pragma omp for schedule(dynamic, 1024)
            for (int e = 0; e < M; e++) {
                ll s = hOff[N + e] + 2, t = hOff[N + e + 1];
                if (t - s <= 2) continue;
                int a = eu[e];
                tmp.clear();
                for (ll p = s; p < t; p += 2) {
                    int f = hAdj[p] - N;
                    int w = eu[f] == a ? ev[f] : eu[f];
                    tmp.push_back({w, {hAdj[p], hAdj[p + 1]}});
                }
                std::sort(tmp.begin(), tmp.end());
                for (size_t k = 0; k < tmp.size(); k++) {
                    hAdj[s + 2 * k] = tmp[k].second.first;
                    hAdj[s + 2 * k + 1] = tmp[k].second.second;
                }
            }
        }
    }
    maxDegH = md;
    return triangles;
}

// ---------------------------------------------------------------------------
// Alternative step 2 (report comparison only): Jones-Plassmann
// - priority = (degree, hash)
// - node colored once all higher-priority neighbors are colored
// - ready nodes never adjacent: frontier colored in parallel
// ---------------------------------------------------------------------------

static int JonesPlassmann(int &rounds) {
    ClockStop();  // allocation: not timed
    std::vector<uint64_t> key(V);
    std::vector<int> cnt(V);
    std::vector<int> bufA(V), bufB(V);
    std::vector<std::vector<int>> marks(nthreads, std::vector<int>(maxDegH + 2, 0));
    ClockStart();
    int *cur = bufA.data(), *nxt = bufB.data();
    int curSize = 0, newSize = 0;
    std::vector<PaddedList> local(nthreads);
    std::vector<ll> offs(nthreads + 1);
    rounds = 0;

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num(), T = omp_get_num_threads();
        std::vector<int> &mine = local[tid].v;
        #pragma omp for schedule(static)
        for (int v = 0; v < V; v++) {
            key[v] = (uint64_t)HDeg(v) << 32 | Hash32(v);
            color[v] = -1;
        }
        auto higher = [&](int a, int b) { return key[a] > key[b] || (key[a] == key[b] && a > b); };
        #pragma omp for schedule(dynamic, 1024)
        for (int v = 0; v < V; v++) {
            int c = 0;
            for (ll i = hOff[v]; i < hOff[v + 1]; i++) c += higher(hAdj[i], v);
            cnt[v] = c;
            if (c == 0) mine.push_back(v);
        }
        std::vector<int> &mark = marks[tid];
        int stamp = 0;
        bool first = true;
        while (true) {
            if (!first) {
                #pragma omp for schedule(dynamic, 64)
                for (int i = 0; i < curSize; i++) {
                    int v = cur[i];
                    color[v] = FirstFit(v, mark, stamp);
                    for (ll j = hOff[v]; j < hOff[v + 1]; j++) {
                        int u = hAdj[j];
                        if (!higher(v, u)) continue;
                        int left;
                        #pragma omp atomic capture
                        left = --cnt[u];
                        if (left == 0) mine.push_back(u);
                    }
                }
            }
            first = false;
            // gather per-thread lists -> next frontier
            #pragma omp single
            {
                offs[0] = 0;
                for (int t = 0; t < T; t++) offs[t + 1] = offs[t] + local[t].v.size();
                newSize = (int)offs[T];
            }
            std::copy(mine.begin(), mine.end(), nxt + offs[tid]);
            mine.clear();
            #pragma omp barrier
            #pragma omp single
            {
                std::swap(cur, nxt);
                curSize = newSize;
                if (curSize) rounds++;
            }
            if (curSize == 0) break;
        }
    }
    int K = 0;
    #pragma omp parallel for reduction(max : K) num_threads(nthreads)
    for (int v = 0; v < V; v++) K = std::max(K, color[v] + 1);
    return K;
}

// ---------------------------------------------------------------------------
// Step 2: batched parallel DSATUR
// - saturation = #distinct colors among colored neighbors
// - each round: uncolored nodes at max saturation L
// - pick those beating all uncolored level-L neighbors (degree, hash)
// - picked nodes never adjacent: colored in parallel, no conflicts
// - deterministic: independent of thread count
// Buckets
// - one per saturation level
// - stale entries (colored / moved up) dropped when scanned
// - colors < 128: per-node bitset; larger colors: recount
// Start / new component
// - no saturated node left: color largest-degree uncolored node alone
// Small rounds
// - bucket < SMALL_ROUND: one thread (too little work for barriers)
// ---------------------------------------------------------------------------

static int BatchedDsatur(ll &rounds) {
    ClockStop();  // allocation + init: not timed
    std::vector<uint64_t> seen(2 * (size_t)V, 0);
    std::vector<int> sat(V, 0), lastLevel(V, 0);
    std::vector<char> touched(V, 0), recount(V, 0);
    std::vector<std::vector<int>> bucket(2);
    std::vector<int> cand(V), picked(V);
    std::vector<PaddedList> local(nthreads), levelPos(nthreads);
    std::vector<ll> offs(nthreads + 1);
    int L = 0, candSize = 0, pickedSize = 0;
    bool done = false;
    // kind of next round: 0 small, 1 large, 2 finished
    // - single has no barrier at its start
    // - fast thread may already be in round r+1
    // - so: double-buffered by round parity
    int kind[2] = {0, 0};
    rounds = 0;

    const int maxDeg = maxDegH;
    std::vector<int> byDeg(V), degStart(maxDeg + 2, 0);
    std::vector<std::vector<int>> marks(nthreads, std::vector<int>(maxDeg + 2, 0));
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int v = 0; v < V; v++) color[v] = HDeg(v) == 0 ? 0 : -1;  // isolated: color 0
    ClockStart();

    // nodes by decreasing degree (counting sort): start new components
    for (int v = 0; v < V; v++) degStart[maxDeg - HDeg(v) + 1]++;
    for (int d = 0; d <= maxDeg; d++) degStart[d + 1] += degStart[d];
    for (int v = 0; v < V; v++) byDeg[degStart[maxDeg - HDeg(v)]++] = v;
    int nextStart = 0;

    auto greater = [&](int a, int b) {  // tie-break: degree, hash, id
        if (HDeg(a) != HDeg(b)) return HDeg(a) > HDeg(b);
        uint32_t ha = Hash32(a), hb = Hash32(b);
        return ha != hb ? ha > hb : a > b;
    };
    // concatenate per-thread lists into dst (all threads call it)
    // - extra(): runs inside the single, after all loops finished
    auto gather = [&](int tid, int T, int *dst, int &size, auto extra) {
        #pragma omp barrier
        #pragma omp single
        {
            offs[0] = 0;
            for (int t = 0; t < T; t++) offs[t + 1] = offs[t] + local[t].v.size();
            size = (int)offs[T];
            extra();
        }
        std::copy(local[tid].v.begin(), local[tid].v.end(), dst + offs[tid]);
        local[tid].v.clear();
        #pragma omp barrier
    };

    auto decide = [&](int r) {  // kind of round r (inside a single)
        kind[r & 1] = done ? 2 : (L >= 1 && bucket[L].size() >= SMALL_ROUND) ? 1 : 0;
    };

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num(), T = omp_get_num_threads();
        std::vector<int> &mine = local[tid].v;
        std::vector<int> &mark = marks[tid];  // colors <= maxDeg
        int stamp = 0;

        auto valid = [&](int v) { return color[v] < 0 && sat[v] == L; };
        auto isPick = [&](int v) {
            for (ll j = hOff[v]; j < hOff[v + 1]; j++) {
                int u = hAdj[j];
                if (color[u] < 0 && sat[u] == L && greater(u, v)) return false;
            }
            return true;
        };
        // record v's new color at its uncolored neighbors
        // - saturation grew: neighbor appended to out (once per round)
        auto raise = [&](int v, std::vector<int> &out) {
            int c = color[v];
            for (ll j = hOff[v]; j < hOff[v + 1]; j++) {
                int u = hAdj[j];
                if (color[u] >= 0) continue;
                bool changed = false;
                if (c < 128) {
                    uint64_t bit = 1ULL << (c & 63), cur;
                    uint64_t &w = seen[2 * (size_t)u + (c >> 6)];
                    #pragma omp atomic read
                    cur = w;
                    if (!(cur & bit)) {  // cheap read first: usually known
                        #pragma omp atomic capture
                        { cur = w; w |= bit; }
                        if (!(cur & bit)) {
                            #pragma omp atomic
                            sat[u]++;
                            changed = true;
                        }
                    }
                } else {
                    #pragma omp atomic write
                    recount[u] = 1;
                    changed = true;
                }
                if (changed) {
                    char was;
                    #pragma omp atomic read
                    was = touched[u];
                    if (!was) {
                        #pragma omp atomic capture
                        { was = touched[u]; touched[u] = 1; }
                        if (!was) out.push_back(u);
                    }
                }
            }
        };
        // finish a raised node; true = goes to a new bucket
        auto settle = [&](int u) {
            touched[u] = 0;
            if (recount[u]) {
                recount[u] = 0;
                ++stamp;
                int s = 0;
                for (ll j = hOff[u]; j < hOff[u + 1]; j++) {
                    int c = color[hAdj[j]];
                    if (c >= 0 && mark[c] != stamp) { mark[c] = stamp; s++; }
                }
                sat[u] = s;
            }
            if (sat[u] == lastLevel[u]) return false;
            lastLevel[u] = sat[u];
            return true;
        };

        for (int r = 0;; r++) {
            int k = kind[r & 1];
            if (k == 2) break;
            if (k == 0) {
                // ---- small round: one thread ----
                #pragma omp single
                {
                    while (L >= 1) {  // drop stale entries, find top level
                        std::vector<int> &B = bucket[L];
                        size_t k = 0;
                        for (int v : B) if (valid(v)) B[k++] = v;
                        B.resize(k);
                        if (k) break;
                        L--;
                    }
                    pickedSize = 0;
                    if (L >= 1 && bucket[L].size() >= SMALL_ROUND) {
                        // large level found: next round runs in parallel
                    } else {
                        if (L < 1) {
                            while (nextStart < V && color[byDeg[nextStart]] >= 0) nextStart++;
                            if (nextStart == V) done = true;
                            else picked[pickedSize++] = byDeg[nextStart];
                        } else {
                            for (int v : bucket[L]) if (isPick(v)) picked[pickedSize++] = v;
                        }
                        for (int i = 0; i < pickedSize; i++) color[picked[i]] = FirstFit(picked[i], mark, stamp);
                        for (int i = 0; i < pickedSize; i++) raise(picked[i], mine);
                        for (int u : mine) {
                            if (!settle(u)) continue;
                            if ((int)bucket.size() <= sat[u]) bucket.resize(sat[u] + 1);
                            bucket[sat[u]].push_back(u);
                            L = std::max(L, sat[u]);
                        }
                        mine.clear();
                        if (!done) rounds++;
                    }
                    decide(r + 1);
                }
                continue;
            }

            // ---- large round: all threads ----
            // (a) valid entries of the top bucket
            {
                const std::vector<int> &B = bucket[L];
                #pragma omp for schedule(static) nowait
                for (size_t i = 0; i < B.size(); i++) if (valid(B[i])) mine.push_back(B[i]);
            }
            gather(tid, T, cand.data(), candSize, [] {});
            if (candSize == 0) {
                #pragma omp single
                { bucket[L].clear(); L--; decide(r + 1); }
                continue;
            }
            // (b) pick local maxima; shrink bucket to valid entries
            #pragma omp for schedule(dynamic, 64) nowait
            for (int i = 0; i < candSize; i++) if (isPick(cand[i])) mine.push_back(cand[i]);
            gather(tid, T, picked.data(), pickedSize,
                   [&] { bucket[L].assign(cand.begin(), cand.begin() + candSize); });
            // (c) color + raise in one pass
            // - independent set: no conflicts
            // - neighbors are never picked: nobody colors them meanwhile
            // - v's list still in cache
            #pragma omp for schedule(dynamic, 64)
            for (int i = 0; i < pickedSize; i++) {
                color[picked[i]] = FirstFit(picked[i], mark, stamp);
                raise(picked[i], mine);
            }
            // (e) raised nodes -> bucket of their new level
            // - count per level, reserve (one thread), copy (all)
            std::vector<int> &pos = levelPos[tid].v;
            int myMax = 0;
            size_t keep = 0;
            for (int u : mine) {
                if (!settle(u)) continue;
                mine[keep++] = u;
                myMax = std::max(myMax, sat[u]);
            }
            mine.resize(keep);
            pos.assign(myMax + 1, 0);
            for (int u : mine) pos[sat[u]]++;
            #pragma omp barrier
            #pragma omp single
            {
                int newMax = 0;
                for (int t = 0; t < T; t++) newMax = std::max(newMax, (int)levelPos[t].v.size() - 1);
                if ((int)bucket.size() <= newMax) bucket.resize(newMax + 1);
                for (int l = 1; l <= newMax; l++) {
                    size_t base = bucket[l].size();
                    for (int t = 0; t < T; t++) {
                        if (l < (int)levelPos[t].v.size()) {
                            size_t c = levelPos[t].v[l];
                            levelPos[t].v[l] = (int)base;
                            base += c;
                        }
                    }
                    bucket[l].resize(base);
                }
                L = std::max(L, newMax);
                rounds++;
                decide(r + 1);
            }
            for (int u : mine) bucket[sat[u]][pos[sat[u]]++] = u;
            mine.clear();
            #pragma omp barrier
        }
    }
    int K = 0;
    #pragma omp parallel for reduction(max : K) num_threads(nthreads)
    for (int v = 0; v < V; v++) K = std::max(K, color[v] + 1);
    return K;
}

// ---------------------------------------------------------------------------
// Step 3: parallel top-class elimination (local search)
// - t = K-1 = highest color; its nodes are independent -> parallel
// - move v to a free color c < t, or
// - move its k <= MAX_BLOCKERS blockers (all color c) to other colors, then v to c
// - fewer blockers tried first
// Conflicts between proposals
// - each claims its changed nodes + neighbors (atomic min of its index)
// - applied only if it owns all its claims
// - smallest proposal always wins: progress every pass
// - class t empty: K - 1, next class
// - deterministic
// ---------------------------------------------------------------------------

static inline void AtomicMin(std::atomic<int> &a, int x) {
    int cur = a.load(std::memory_order_relaxed);
    while (x < cur && !a.compare_exchange_weak(cur, x, std::memory_order_relaxed)) {}
}

static int EliminateTopClasses(int K, int lowerBound, ll &passes) {
    ClockStop();  // allocation + init: not timed
    std::unique_ptr<std::atomic<int>[]> owner(new std::atomic<int>[V]);
    std::vector<int> top, propC;
    std::vector<std::vector<std::pair<int, int>>> moves;  // (w, d) moves per proposal
    std::vector<char> accepted;
    top.reserve(V);
    passes = 0;
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int v = 0; v < V; v++) owner[v].store(INT32_MAX, std::memory_order_relaxed);
    ClockStart();

    auto claim = [&](int x, int id) {
        AtomicMin(owner[x], id);
        for (ll j = hOff[x]; j < hOff[x + 1]; j++) AtomicMin(owner[hAdj[j]], id);
    };
    auto owns = [&](int x, int id) {
        if (owner[x].load(std::memory_order_relaxed) != id) return false;
        for (ll j = hOff[x]; j < hOff[x + 1]; j++)
            if (owner[hAdj[j]].load(std::memory_order_relaxed) != id) return false;
        return true;
    };
    auto release = [&](int x) {
        owner[x].store(INT32_MAX, std::memory_order_relaxed);
        for (ll j = hOff[x]; j < hOff[x + 1]; j++) owner[hAdj[j]].store(INT32_MAX, std::memory_order_relaxed);
    };

    while (K > lowerBound && K > 1) {
        const int t = K - 1;
        top.clear();
        for (int v = 0; v < V; v++) if (color[v] == t) top.push_back(v);
        bool stuck = false;
        while (!top.empty()) {
            passes++;
            int P = (int)top.size(), acceptedCount = 0, proposals = 0;
            ClockStop();  // init proposal arrays: not timed
            propC.assign(P, -1);
            moves.assign(P, {});
            accepted.assign(P, 0);
            ClockStart();
            #pragma omp parallel num_threads(nthreads) reduction(+ : acceptedCount, proposals)
            {
                std::vector<int> cnt(t + 1), st(t + 1, 0), used(t + 1, 0);
                std::vector<std::pair<int, int>> tryMoves;
                int stamp = 0, stamp2 = 0;
                #pragma omp for schedule(dynamic, 16)
                for (int i = 0; i < P; i++) {
                    int v = top[i];
                    ++stamp;
                    for (ll j = hOff[v]; j < hOff[v + 1]; j++) {
                        int u = hAdj[j], c = color[u];
                        if (c >= t) continue;
                        if (st[c] != stamp) { st[c] = stamp; cnt[c] = 1; }
                        else cnt[c]++;
                    }
                    for (int c = 0; c < t && propC[i] < 0; c++)
                        if (st[c] != stamp) propC[i] = c;                 // free color
                    for (int k = 1; k <= MAX_BLOCKERS && propC[i] < 0; k++) {
                        for (int c = 0; c < t && propC[i] < 0; c++) {
                            if (cnt[c] != k) continue;                    // exactly k blockers
                            tryMoves.clear();
                            for (ll j = hOff[v]; j < hOff[v + 1]; j++) {
                                int w = hAdj[j];
                                if (color[w] != c) continue;
                                ++stamp2;
                                for (ll q = hOff[w]; q < hOff[w + 1]; q++) {
                                    int x = hAdj[q];
                                    if (x != v && color[x] < t) used[color[x]] = stamp2;
                                }
                                int d = 0;
                                while (d < t && (d == c || used[d] == stamp2)) d++;
                                if (d == t) break;                        // w cannot move
                                tryMoves.push_back({w, d});
                            }
                            if ((int)tryMoves.size() == k) { propC[i] = c; moves[i] = tryMoves; }
                        }
                    }
                    if (propC[i] >= 0) {
                        proposals++;
                        claim(v, i);
                        for (auto &m : moves[i]) claim(m.first, i);
                    }
                }
                #pragma omp for schedule(dynamic, 16)
                for (int i = 0; i < P; i++) {
                    if (propC[i] < 0) continue;
                    bool ok = owns(top[i], i);
                    for (auto &m : moves[i]) ok = ok && owns(m.first, i);
                    accepted[i] = ok;
                }
                #pragma omp for schedule(dynamic, 16)
                for (int i = 0; i < P; i++) {
                    if (!accepted[i]) continue;
                    for (auto &m : moves[i]) color[m.first] = m.second;
                    color[top[i]] = propC[i];
                    acceptedCount++;
                }
                #pragma omp for schedule(dynamic, 16)
                for (int i = 0; i < P; i++) {
                    if (propC[i] < 0) continue;
                    release(top[i]);
                    for (auto &m : moves[i]) release(m.first);
                }
            }
            if (acceptedCount == 0) {  // rest of class t cannot move
                stuck = true;
                break;
            }
            size_t keep = 0;
            for (int v : top) if (color[v] == t) top[keep++] = v;
            top.resize(keep);
        }
        if (stuck) break;
        K--;  // class t is empty
    }
    return K;
}

// ---------------------------------------------------------------------------
// Lower bound = size of a large clique of H
// - greedy, from the highest-degree nodes, in parallel
// - any edge: >= 3 colors (u, v, edge)
// ---------------------------------------------------------------------------

static int CliqueLowerBound() {
    if (V == 0) return 0;
    int best = (M > 0) ? 3 : 1;
    int S = std::min(V, 2048);
    ClockStop();  // allocation + init: not timed
    std::vector<int> idx(V);
    for (int v = 0; v < V; v++) idx[v] = v;
    // calloc: zeroed lazily, only touched pages cost
    std::vector<int *> marks(nthreads);
    for (int t = 0; t < nthreads; t++) marks[t] = (int *)calloc(V, sizeof(int));
    ClockStart();
    std::partial_sort(idx.begin(), idx.begin() + S, idx.end(),
                      [](int a, int b) { return HDeg(a) > HDeg(b); });
    #pragma omp parallel num_threads(nthreads) reduction(max : best)
    {
        int *mark = marks[omp_get_thread_num()];
        std::vector<int> cand, next;
        int stamp = 0;
        #pragma omp for schedule(dynamic, 1)
        for (int si = 0; si < S; si++) {
            int s = idx[si], size = 1;
            cand.assign(hAdj.begin() + hOff[s], hAdj.begin() + hOff[s + 1]);
            while (!cand.empty()) {
                int c = cand[0];
                for (int x : cand) if (HDeg(x) > HDeg(c)) c = x;
                size++;
                ++stamp;  // keep candidates adjacent to c
                for (ll i = hOff[c]; i < hOff[c + 1]; i++) mark[hAdj[i]] = stamp;
                next.clear();
                for (int x : cand) if (mark[x] == stamp) next.push_back(x);
                cand.swap(next);
            }
            best = std::max(best, size);
        }
    }
    ClockStop();
    for (int t = 0; t < nthreads; t++) free(marks[t]);
    ClockStart();
    return best;
}

// ---------------------------------------------------------------------------
// Step 4 (small graphs only): exact DSATUR branch and bound
// ---------------------------------------------------------------------------

struct ExactSearch {
    int best, lb;
    ll nodes = 0;
    bool aborted = false;
    std::vector<int> col, satDeg, bestCol;
    std::vector<std::vector<int>> satCnt;  // [v][c] = #neighbors of v with color c

    void Assign(int v, int c, int delta) {
        for (ll i = hOff[v]; i < hOff[v + 1]; i++) {
            int u = hAdj[i];
            int before = satCnt[u][c];
            satCnt[u][c] += delta;
            if (before == 0 && delta > 0) satDeg[u]++;
            if (before == 1 && delta < 0) satDeg[u]--;
        }
    }
    void Dfs(int colored, int used) {
        if (aborted || best <= lb || used >= best) return;
        if (++nodes > EXACT_BUDGET) { aborted = true; return; }
        if (colored == V) { best = used; bestCol = col; return; }
        int v = -1;
        for (int x = 0; x < V; x++)
            if (col[x] < 0 && (v < 0 || satDeg[x] > satDeg[v] || (satDeg[x] == satDeg[v] && HDeg(x) > HDeg(v))))
                v = x;
        for (int c = 0; c <= used; c++) {
            if (c == used && used + 1 >= best) break;  // new color can't beat best
            if (satCnt[v][c]) continue;
            col[v] = c;
            Assign(v, c, +1);
            Dfs(colored + 1, std::max(used, c + 1));
            Assign(v, c, -1);
            col[v] = -1;
            if (aborted || best <= lb) return;
        }
    }
};

// true = search finished (result optimal)
static bool ExactColor(int &K, int lowerBound) {
    ClockStop();  // allocation: not timed
    ExactSearch s;
    s.best = K;
    s.lb = lowerBound;
    s.col.assign(V, -1);
    s.satDeg.assign(V, 0);
    s.satCnt.assign(V, std::vector<int>(K + 1, 0));
    ClockStart();
    s.Dfs(0, 0);
    if (!s.bestCol.empty()) { color = s.bestCol; K = s.best; }
    return !s.aborted;
}

// ---------------------------------------------------------------------------

static void WriteOutput(const char *fname, int K) {
    FILE *f = fopen(fname, "w");
    if (!f) { fprintf(stderr, "cannot open %s\n", fname); exit(1); }
    std::vector<char> out;
    out.reserve((size_t)N * 16 + (size_t)M * 28 + 32);
    char tmp[64];
    for (int u = 0; u < N; u++) {
        int len = snprintf(tmp, sizeof tmp, "%d, %d\n", u, color[u]);
        out.insert(out.end(), tmp, tmp + len);
    }
    for (int e = 0; e < M; e++) {
        int len = snprintf(tmp, sizeof tmp, "(%d, %d), %d\n", eu[e], ev[e], color[N + e]);
        out.insert(out.end(), tmp, tmp + len);
    }
    int len = snprintf(tmp, sizeof tmp, "%d\n", K);
    out.insert(out.end(), tmp, tmp + len);
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <input file> <output file> <number of threads>\n", argv[0]);
        return 1;
    }
    nthreads = std::max(1, atoi(argv[3]));
    omp_set_num_threads(nthreads);
    ReadInput(argv[1]);
    color.assign(N + M, -1);

    // timed phases (allocations inside are excluded)
    ClockStart();
    ll triangles = BuildConflictGraph();
    ClockStop();
    double t1 = measured;
    // Q1_INIT=jp: Jones-Plassmann instead of DSATUR (report comparison only)
    const char *init = getenv("Q1_INIT");
    ll rounds = 0;
    int K;
    ClockStart();
    if (init && !strcmp(init, "jp")) { int r; K = JonesPlassmann(r); rounds = r; }
    else K = BatchedDsatur(rounds);
    ClockStop();
    int K0 = K;
    double t2 = measured;
    ClockStart();
    int lb = CliqueLowerBound();
    ll passes = 0;
    K = EliminateTopClasses(K, lb, passes);
    ClockStop();
    double t3 = measured;
    bool optimal = (K == lb);
    ClockStart();
    if (!optimal && V <= EXACT_MAX_NODES) optimal = ExactColor(K, lb);
    ClockStop();
    double t0 = 0.0, t4 = measured;

    // color ids exactly 0..K-1
    std::vector<int> remap(K, 0);
    for (int v = 0; v < V; v++) remap[color[v]] = 1;
    int used = 0;
    for (int c = 0; c < K; c++) remap[c] = remap[c] ? used++ : -1;
    for (int v = 0; v < V; v++) color[v] = remap[color[v]];
    K = used;

    fprintf(stderr,
            "n = %d, m = %d, triangles = %lld, |E(H)| = %lld, threads = %d\n"
            "colors: initial = %d, final = %d, lower bound = %d%s\n"
            "rounds: coloring = %lld, elimination passes = %lld\n"
            "time (computation only, allocation/initialization excluded): "
            "build H %.4f s, coloring %.4f s, improve %.4f s, exact %.4f s, total %.4f s\n",
            N, M, triangles, (ll)(V ? hOff[V] / 2 : 0), nthreads, K0, K, lb,
            optimal ? " (optimal)" : "", rounds, passes,
            t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0);
    WriteOutput(argv[2], K);
    return 0;
}
