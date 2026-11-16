// CS433 Assignment #1, Question 2 (OpenMP)
// Solve Ax = y, A = (2n+1)x(2n+1) "hourglass" matrix
// Usage: ./q2 <input file> <output file> <number of threads>
//
// Structure of A (rows/cols 0..2n)
// - row i non-zero between columns min(i, 2n-i) and max(i, 2n-i)
// - row n: one entry
// - rows n-k, n+k: columns n-k .. n+k
// - solve from the middle outwards
// - step k: 2 new unknowns x[n-k], x[n+k] -> 2x2 system
//
// Storage ("rings")
// - ring k = columns n-k, n+k of every row pair p >= k (rows n-p, n+p)
// - 4 values per (k, p):
//     [0] = A[n-p][n-k]   [1] = A[n-p][n+k]
//     [2] = A[n+p][n-k]   [3] = A[n+p][n+k]
// - step k = one contiguous pass over ring k
// - ring 0: only slots [0], [2] used (column n)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <atomic>
#include <vector>
#include <algorithm>
#include <omp.h>

static int n;               // matrix size 2n+1
static double *R;           // rings, 2(n+1)(n+2) doubles
static size_t *ringOff;     // start of ring k in R
static double *yL, *yR;     // yL[p] = y[n-p], yR[p] = y[n+p]
static double *xL, *xR;     // xL[p] = x[n-p], xR[p] = x[n+p]
static double *xTrue;       // known solution (InitializeInput only)

static const int SERIAL_BLOCK = 8;             // steps per pass (1 thread)
static const int MIN_ROWS_PER_THREAD = 500;    // min row pairs per thread

static inline double &entry(int k, int p, int slot) {
    return R[ringOff[k] + 4 * (size_t)(p - k) + slot];
}

static void Allocate() {
    ringOff = (size_t *)malloc((n + 2) * sizeof(size_t));
    size_t off = 0;
    for (int k = 0; k <= n; k++) { ringOff[k] = off; off += 4 * (size_t)(n - k + 1); }
    ringOff[n + 1] = off;
    R = (double *)calloc(off, sizeof(double));
    yL = (double *)calloc(n + 1, sizeof(double));
    yR = (double *)calloc(n + 1, sizeof(double));
    xL = (double *)calloc(n + 1, sizeof(double));
    xR = (double *)calloc(n + 1, sizeof(double));
    if (!R || !yL || !yR || !xL || !xR) { fprintf(stderr, "out of memory\n"); exit(1); }
}

// store A[r][c] (c inside row r's non-zero range)
static inline void SetA(int r, int c, double v) {
    int p = abs(r - n), k = abs(c - n);
    entry(k, p, 2 * (r > n) + (c > n)) = v;
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

// next number in the buffer (or nullptr)
// - anything else = separator (spaces, commas, newlines, brackets)
// - UTF-8 minus sign (U+2212, as in the PDF) accepted
static char *NextNumber(char *s, char *end, bool &neg) {
    neg = false;
    while (s < end) {
        unsigned char ch = (unsigned char)*s;
        if ((ch >= '0' && ch <= '9') || ch == '-' || ch == '+' || ch == '.') return s;
        if (ch == 0xE2 && s + 2 < end && (unsigned char)s[1] == 0x88 && (unsigned char)s[2] == 0x92) {
            neg = true;
            return s + 3;
        }
        s++;
    }
    return nullptr;
}

static void ReadInput(const char *fname) {
    FILE *f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", fname); exit(1); }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(size + 1);
    size_t got = fread(buf, 1, size, f);
    buf[got] = '\0';
    fclose(f);
    char *s = buf, *end = buf + got;

    bool neg;
    auto next = [&]() -> double {
        s = NextNumber(s, end, neg);
        if (!s) { fprintf(stderr, "input file ended early\n"); exit(1); }
        char *e;
        double v = strtod(s, &e);
        if (e == s) { fprintf(stderr, "bad number in input\n"); exit(1); }
        s = e;
        return neg ? -v : v;
    };

    n = (int)next();
    if (n < 0) { fprintf(stderr, "bad n\n"); exit(1); }
    Allocate();
    int N = 2 * n + 1;
    for (int r = 0; r < N; r++) {
        int lo = r <= n ? r : 2 * n - r, hi = 2 * n - lo;
        for (int c = lo; c <= hi; c++) SetA(r, c, next());
    }
    for (int r = 0; r < N; r++) {
        double v = next();
        if (r <= n) yL[n - r] = v; else yR[r - n] = v;
    }
    free(buf);
}

// Random test case of size 2n+1 (quick testing, no input file)
// - A and known x generated first, then y = Ax
// - 2x2 pivot blocks dominant: numerically stable
// - other non-zeros in [-1, 1]
[[maybe_unused]] static void InitializeInput(int size_n) {
    n = size_n;
    Allocate();
    int N = 2 * n + 1;
    xTrue = (double *)malloc(N * sizeof(double));
    std::vector<double> y(N, 0.0);
    unsigned long long st = 12345;
    auto rnd = [&]() {  // uniform in [0, 1)
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        return (double)(st >> 11) * (1.0 / 9007199254740992.0);
    };
    for (int i = 0; i < N; i++) xTrue[i] = 2.0 * rnd() - 1.0;
    double big = 2.0 * sqrt(2.0 * n + 1.0) + 4.0;
    for (int r = 0; r < N; r++) {
        int lo = r <= n ? r : 2 * n - r, hi = 2 * n - lo;
        double sum = 0.0;
        for (int c = lo; c <= hi; c++) {
            double v;
            if (c == r) v = (rnd() < 0.5 ? -1 : 1) * big * (1.0 + rnd());  // pivot block diagonal
            else v = 2.0 * rnd() - 1.0;
            SetA(r, c, v);
            sum += v * xTrue[c];
        }
        y[r] = sum;
    }
    for (int r = 0; r < N; r++) {
        if (r <= n) yL[n - r] = y[r]; else yR[r - n] = y[r];
    }
}

// ---------------------------------------------------------------------------
// Solvers
// ---------------------------------------------------------------------------

// solve the 2x2 system of step k (1x1 for k = 0)
// - yL[k], yR[k]: inner contributions already subtracted
static inline void SolvePivot(int k, double &xl, double &xr) {
    if (k == 0) {
        xl = yL[0] / entry(0, 0, 0);
        xr = 0.0;  // unused: ring 0 slots [1], [3] are zero
        return;
    }
    double a = entry(k, k, 0), b = entry(k, k, 1);
    double c = entry(k, k, 2), d = entry(k, k, 3);
    double det = a * d - b * c;
    xl = (d * yL[k] - b * yR[k]) / det;
    xr = (a * yR[k] - c * yL[k]) / det;
}

// subtract step k's contribution from row pairs [from, to)
static inline void Update(int k, int from, int to, double xl, double xr) {
    const double *e = R + ringOff[k] + 4 * (size_t)(from - k);
    double *__restrict yl = yL, *__restrict yr = yR;
    #pragma omp simd
    for (int p = from; p < to; p++) {
        const double *q = e + 4 * (size_t)(p - from);
        yl[p] -= q[0] * xl + q[1] * xr;
        yr[p] -= q[2] * xl + q[3] * xr;
    }
}

// apply steps k0 .. k0+cnt-1 to row pairs [from, to) in one pass
// - y loaded/stored once instead of cnt times
// - same operations, same order per row: bit-identical
static inline void UpdateMulti(int k0, int cnt, int from, int to) {
    double *__restrict yl = yL, *__restrict yr = yR;
    for (int p = from; p < to; p++) {
        double a = yl[p], b = yr[p];
        for (int s = 0; s < cnt; s++) {
            int k = k0 + s;
            const double *q = R + ringOff[k] + 4 * (size_t)(p - k);
            a -= q[0] * xL[k] + q[1] * xR[k];
            b -= q[2] * xL[k] + q[3] * xR[k];
        }
        yl[p] = a;
        yr[p] = b;
    }
}

// Single thread: blocks of SERIAL_BLOCK steps
// - solve the block's pivots (updating only the block's rows)
// - then outer rows get all the block's steps in one pass
// - measured: 12-18% faster
// - not used in parallel: too many memory streams at once
static void SolveSerial() {
    for (int k0 = 0; k0 <= n; k0 += SERIAL_BLOCK) {
        int end = std::min(k0 + SERIAL_BLOCK, n + 1);
        for (int k = k0; k < end; k++) {
            SolvePivot(k, xL[k], xR[k]);
            Update(k, k + 1, end, xL[k], xR[k]);
        }
        UpdateMulti(k0, end - k0, end, n + 1);
    }
}

// Version 1: barrier per step
// - every thread solves the pivot itself (no extra sync)
// - outer row pairs split statically
static void SolveBarrier(int nthreads) {
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        for (int k = 0; k <= n; k++) {
            double xl, xr;
            SolvePivot(k, xl, xr);
            if (tid == 0) { xL[k] = xl; xR[k] = xr; }
            int cnt = n - k, T = omp_get_num_threads();
            int from = k + 1 + (int)((long long)cnt * tid / T);
            int to = k + 1 + (int)((long long)cnt * (tid + 1) / T);
            Update(k, from, to, xl, xr);
            #pragma omp barrier
        }
    }
}

static inline void CpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

// Version 2 (default): pipelined, no barriers
// - one fixed block of row pairs per thread
// - row pair p gets p updates: boundaries at n*sqrt(t/T) balance the work
// - step k applied as soon as x of step k is published
// - own pivots: next pivot row first, published right away (look-ahead)
// - pivots published in order via one atomic counter (release/acquire)
//
// block boundaries (computed before the timed solve)
static std::vector<int> PipelineBounds(int nthreads) {
    std::vector<int> bound(nthreads + 1);
    for (int t = 0; t <= nthreads; t++)
        bound[t] = (t == nthreads) ? n + 1 : (int)(n * sqrt((double)t / nthreads));
    for (int t = 1; t <= nthreads; t++)
        if (bound[t] < bound[t - 1]) bound[t] = bound[t - 1];
    return bound;
}

static void SolvePipelined(int nthreads, const std::vector<int> &bound) {
    std::atomic<int> solved(0);  // pivots 0 .. solved-1 ready in xL/xR

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int lo = bound[tid], hi = bound[tid + 1];
        if (lo == hi) hi = 0;  // empty block
        for (int k = 0; k < hi; k++) {
            if (k < lo) {
                while (solved.load(std::memory_order_acquire) <= k) CpuRelax();
                Update(k, lo, hi, xL[k], xR[k]);
            } else {
                if (k == lo) {  // first own pivot: row already fully updated
                    SolvePivot(k, xL[k], xR[k]);
                    solved.store(k + 1, std::memory_order_release);
                }
                double xl = xL[k], xr = xR[k];
                if (k + 1 < hi) {  // look-ahead: row k+1, publish x of step k+1
                    Update(k, k + 1, k + 2, xl, xr);
                    SolvePivot(k + 1, xL[k + 1], xR[k + 1]);
                    solved.store(k + 2, std::memory_order_release);
                }
                Update(k, k + 2, hi, xl, xr);
            }
        }
    }
}

// ---------------------------------------------------------------------------

static void WriteOutput(const char *fname) {
    FILE *f = fopen(fname, "w");
    if (!f) { fprintf(stderr, "cannot open %s\n", fname); exit(1); }
    int N = 2 * n + 1;
    std::vector<char> line;
    line.reserve((size_t)N * 24 + 2);
    char tmp[64];
    for (int i = 0; i < N; i++) {
        double v = i <= n ? xL[n - i] : xR[i - n];
        int len = snprintf(tmp, sizeof tmp, i ? " %.15g" : "%.15g", v);
        line.insert(line.end(), tmp, tmp + len);
    }
    line.push_back('\n');
    fwrite(line.data(), 1, line.size(), f);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <input file> <output file> <number of threads>\n", argv[0]);
        return 1;
    }
    int nthreads = atoi(argv[3]);
    if (nthreads < 1) nthreads = 1;

    ReadInput(argv[1]);
    // quick testing on large matrices:
    // - comment out ReadInput above, uncomment below
    // - argv[1] is then n
    // InitializeInput(atoi(argv[1]));

    // at most one thread per MIN_ROWS_PER_THREAD row pairs
    // - less work per step: sync costs more than the thread gains
    // - memory bandwidth already saturated by a few threads
    int requested = nthreads;
    nthreads = std::max(1, std::min(nthreads, (n + 1) / MIN_ROWS_PER_THREAD));

    // pipelined spin-waits: needs one core per thread
    // - more threads than cores: barrier version
    // - Q2_ALGO=barrier|serial: force a version (experiments)
    const char *algo = getenv("Q2_ALGO");
    bool useBarrier = (algo && !strcmp(algo, "barrier")) || nthreads > omp_get_num_procs();
    bool useSerial = (algo && !strcmp(algo, "serial")) || n < 64;

    std::vector<int> bound = PipelineBounds(nthreads);  // setup: not timed

    // timed: the solve only
    double t0 = omp_get_wtime();
    if (useSerial || nthreads == 1) SolveSerial();
    else if (useBarrier) SolveBarrier(nthreads);
    else SolvePipelined(nthreads, bound);
    double t1 = omp_get_wtime();

    fprintf(stderr, "n = %d, threads = %d (used %d), solve time = %.6f s\n", n, requested, nthreads, t1 - t0);
    if (xTrue) {
        double err = 0.0;
        for (int i = 0; i <= 2 * n; i++) {
            double v = i <= n ? xL[n - i] : xR[i - n];
            err = fmax(err, fabs(v - xTrue[i]));
        }
        fprintf(stderr, "max |x - x_true| = %.3e\n", err);
    }
    WriteOutput(argv[2]);
    return 0;
}
