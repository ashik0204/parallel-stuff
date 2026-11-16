// Graph generator for Q1 (writes the assignment's input format).
//   ./gen_graph random  n m t seed file   : m random edges + t random triangles
//   ./gen_graph cluster n k p mout seed file : clusters of k vertices, each
//                                            intra-cluster pair with prob p, plus mout random edges
//   ./gen_graph powerlaw n m pt seed file  : Holme-Kim model (preferential
//                                            attachment, m edges per new vertex,
//                                            triad formation with prob pt)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
typedef unsigned long long u64;
static u64 st;
static u64 rnd() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; }
static double urand() { return (rnd() >> 11) * (1.0 / 9007199254740992.0); }
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "see source for usage\n"); return 1; }
    std::vector<u64> keys;
    int n = atoi(argv[2]);
    auto add = [&](int a, int b) { if (a == b) return; if (a > b) std::swap(a, b); keys.push_back((u64)a << 32 | (unsigned)b); };
    const char *file;
    if (!strcmp(argv[1], "random")) {
        long long m = atoll(argv[3]), t = atoll(argv[4]);
        st = atoll(argv[5]) * 2654435761ULL + 1; file = argv[6];
        for (long long i = 0; i < m; i++) add(rnd() % n, rnd() % n);
        for (long long i = 0; i < t; i++) { int a = rnd() % n, b = rnd() % n, c = rnd() % n; add(a, b); add(b, c); add(a, c); }
    } else if (!strcmp(argv[1], "cluster")) {
        int k = atoi(argv[3]); double p = atof(argv[4]); long long mout = atoll(argv[5]);
        st = atoll(argv[6]) * 2654435761ULL + 1; file = argv[7];
        for (int s = 0; s < n; s += k)
            for (int a = s; a < std::min(n, s + k); a++)
                for (int b = a + 1; b < std::min(n, s + k); b++)
                    if (urand() < p) add(a, b);
        for (long long i = 0; i < mout; i++) add(rnd() % n, rnd() % n);
    } else if (!strcmp(argv[1], "powerlaw")) {
        int m = atoi(argv[3]); double pt = atof(argv[4]);
        st = atoll(argv[5]) * 2654435761ULL + 1; file = argv[6];
        std::vector<std::vector<int>> adj(n);
        std::vector<int> ends;  // endpoint list: preferential choice = uniform pick
        for (int a = 0; a <= m; a++) for (int b = a + 1; b <= m; b++) {
            adj[a].push_back(b); adj[b].push_back(a); ends.push_back(a); ends.push_back(b); add(a, b);
        }
        for (int v = m + 1; v < n; v++) {
            int prev = -1;
            std::vector<int> chosen;
            for (int j = 0; j < m; j++) {
                int w;
                if (prev >= 0 && urand() < pt && !adj[prev].empty()) w = adj[prev][rnd() % adj[prev].size()];
                else w = ends[rnd() % ends.size()];
                if (w == v || std::find(chosen.begin(), chosen.end(), w) != chosen.end()) { j--; prev = -1; continue; }
                chosen.push_back(w);
                prev = w;
            }
            for (int w : chosen) { adj[v].push_back(w); adj[w].push_back(v); ends.push_back(v); ends.push_back(w); add(v, w); }
        }
    } else { fprintf(stderr, "unknown type\n"); return 1; }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    FILE *f = fopen(file, "w");
    fprintf(f, "%d\n", n);
    size_t i = 0;
    for (int u = 0; u < n; u++) {
        bool first = true;
        while (i < keys.size() && (int)(keys[i] >> 32) == u) {
            fprintf(f, first ? "%u" : " %u", (unsigned)(keys[i] & 0xffffffffu));
            first = false; i++;
        }
        fprintf(f, "\n");
    }
    fclose(f);
    fprintf(stderr, "n = %d, m = %zu\n", n, keys.size());
    return 0;
}
