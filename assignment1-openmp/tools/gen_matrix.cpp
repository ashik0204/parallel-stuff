// Test generator for Q2: writes an input file (n, non-zeros of A row-wise, y)
// and the known solution x. A and x are generated first and y = Ax.
// Usage: ./gen_matrix n input_file x_file [seed]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s n input_file x_file [seed]\n", argv[0]); return 1; }
    int n = atoi(argv[1]), N = 2 * n + 1;
    unsigned long long st = argc > 4 ? atoll(argv[4]) : 7;
    auto rnd = [&]() {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        return (double)(st >> 11) * (1.0 / 9007199254740992.0);
    };
    std::vector<double> x(N), y(N, 0.0);
    for (auto &v : x) v = 2.0 * rnd() - 1.0;
    double big = 2.0 * sqrt(2.0 * n + 1.0) + 4.0;
    FILE *f = fopen(argv[2], "w");
    fprintf(f, "%d\n", n);
    for (int r = 0; r < N; r++) {
        int lo = r <= n ? r : 2 * n - r, hi = 2 * n - lo;
        for (int c = lo; c <= hi; c++) {
            double v = (c == r) ? (rnd() < 0.5 ? -1 : 1) * big * (1.0 + rnd()) : 2.0 * rnd() - 1.0;
            fprintf(f, c == lo ? "%.17g" : " %.17g", v);
            y[r] += v * x[c];
        }
        fprintf(f, "\n");
    }
    for (int r = 0; r < N; r++) fprintf(f, r ? " %.17g" : "%.17g", y[r]);
    fprintf(f, "\n");
    fclose(f);
    FILE *g = fopen(argv[3], "w");
    for (int r = 0; r < N; r++) fprintf(g, r ? " %.17g" : "%.17g", x[r]);
    fprintf(g, "\n");
    fclose(g);
    return 0;
}
