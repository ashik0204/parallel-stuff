# Compares two one-line vectors (Q2 output vs known x): prints max abs error.
import sys
a = [float(t) for t in open(sys.argv[1]).read().split()]
b = [float(t) for t in open(sys.argv[2]).read().split()]
assert len(a) == len(b), (len(a), len(b))
print("n =", len(a), " max abs err =", max(abs(p - q) for p, q in zip(a, b)))
