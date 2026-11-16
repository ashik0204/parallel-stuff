// Independent checker for Q1 output. Verifies: every vertex and edge colored
// exactly once, rules (i)-(iii), colors are 0..K-1 and the last line is K.
// Usage: ./check_coloring input_file output_file
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <fstream>
#include <sstream>
typedef long long ll;
int main(int argc, char **argv) {
    std::ifstream in(argv[1]);
    std::string line;
    std::getline(in, line);
    int n = atoi(line.c_str());
    std::vector<std::vector<int>> adj(n);
    std::vector<std::pair<int,int>> edges;
    for (int k = 0; k < n && std::getline(in, line); k++) {
        for (char &c : line) if (c < '0' || c > '9') c = ' ';
        std::istringstream ss(line);
        int j;
        while (ss >> j) if (j != k && j < n) edges.push_back({std::min(k, j), std::max(k, j)});
    }
    // a pair listed twice (or in both directions) is one edge
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    for (auto [u, v] : edges) { adj[u].push_back(v); adj[v].push_back(u); }
    for (auto &a : adj) std::sort(a.begin(), a.end());
    std::unordered_map<ll,int> ecol;
    std::vector<int> vcol(n, -1);
    std::ifstream out(argv[2]);
    std::vector<std::string> lines;
    while (std::getline(out, line)) if (!line.empty()) lines.push_back(line);
    if (lines.size() != (size_t)n + edges.size() + 1) { printf("FAIL: %zu lines, expected %zu\n", lines.size(), n + edges.size() + 1); return 1; }
    int maxc = -1;
    for (size_t i = 0; i + 1 < lines.size(); i++) {
        int u, v, c;
        if (lines[i][0] == '(') {
            if (sscanf(lines[i].c_str(), "(%d, %d), %d", &u, &v, &c) != 3) { printf("FAIL parse: %s\n", lines[i].c_str()); return 1; }
            if (u > v) std::swap(u, v);
            ll key = (ll)u * n + v;
            if (ecol.count(key)) { printf("FAIL: edge twice\n"); return 1; }
            ecol[key] = c;
        } else {
            if (sscanf(lines[i].c_str(), "%d, %d", &u, &c) != 2 || u < 0 || u >= n || vcol[u] >= 0) { printf("FAIL parse: %s\n", lines[i].c_str()); return 1; }
            vcol[u] = c;
        }
        if (c < 0) { printf("FAIL: negative color\n"); return 1; }
        maxc = std::max(maxc, c);
    }
    int K = atoi(lines.back().c_str());
    auto E = [&](int a, int b) { if (a > b) std::swap(a, b); auto it = ecol.find((ll)a * n + b); return it == ecol.end() ? -1 : it->second; };
    std::vector<char> used(maxc + 1, 0);
    for (int u = 0; u < n; u++) { if (vcol[u] < 0) { printf("FAIL: vertex %d missing\n", u); return 1; } used[vcol[u]] = 1; }
    ll tri = 0;
    for (auto [u, v] : edges) {
        int c = E(u, v);
        if (c < 0) { printf("FAIL: edge (%d,%d) missing\n", u, v); return 1; }
        used[c] = 1;
        if (vcol[u] == vcol[v]) { printf("FAIL rule i: (%d,%d)\n", u, v); return 1; }
        if (c == vcol[u] || c == vcol[v]) { printf("FAIL rule ii: (%d,%d)\n", u, v); return 1; }
        // triangles (u,v,w) with w > v > u: check rule iii
        int a = std::min(u, v), b = std::max(u, v);
        auto &A = adj[a], &B = adj[b];
        size_t i = 0, j = 0;
        while (i < A.size() && j < B.size()) {
            if (A[i] < B[j]) i++; else if (A[i] > B[j]) j++;
            else {
                int w = A[i];
                if (w > b) {
                    tri++;
                    int c1 = c, c2 = E(a, w), c3 = E(b, w);
                    if (c1 == c2 || c1 == c3 || c2 == c3) { printf("FAIL rule iii: (%d,%d,%d)\n", a, b, w); return 1; }
                }
                i++; j++;
            }
        }
    }
    int distinct = 0;
    for (char x : used) distinct += x;
    if (distinct != maxc + 1 || K != distinct) { printf("FAIL: K=%d, distinct=%d, max=%d\n", K, distinct, maxc); return 1; }
    printf("OK: n=%d m=%zu triangles=%lld colors=%d\n", n, edges.size(), tri, K);
    return 0;
}
