#!/usr/bin/env python3
"""Compute trigram-branch missing continuation mass M(f) from a tokenized shard.

Continuations of a trigram context are 4-grams; 4-gram counts are not in
freq_index.npz, so this derives them directly from the token stream.
Run on a machine that has the tokenized shard (cluster):
    python3 compute_fourgram_missing_mass.py \
        --shard data/tokenized/shard_00001.bin --index data/freq_index.npz \
        --out docs/figs/theory/theory_missing_mass_trigram.csv
Outputs CSV: f, n_contexts, N1_types, Ntypes, M(f)=N1/(f*n), S_eff=Ntypes/n.
Diagnostic-only; does not touch training semantics.
"""
import argparse
import csv

import numpy as np

V = 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    toks = np.memmap(a.shard, dtype=np.uint16, mode="r")
    t = np.asarray(toks, dtype=np.int64)
    print("tokens:", t.size)
    four = ((t[:-3] * V + t[1:-2]) * V + t[2:-1]) * V + t[3:]
    keys, counts = np.unique(four, return_counts=True)
    print("distinct 4-grams:", keys.size)
    prefix = keys // V  # trigram context of each 4-gram type

    z = np.load(a.index)
    tri_keys = z["trigram_keys"].astype(np.int64)
    tri_counts = z["trigram_counts"].astype(np.int64)
    pos = np.searchsorted(tri_keys, prefix)
    ok = (pos < tri_keys.size) & (tri_keys[np.minimum(pos, tri_keys.size - 1)] == prefix)
    print("prefix match rate:", ok.mean())
    f_of_type = tri_counts[pos[ok]]
    c_of_type = counts[ok]

    fmax = int(tri_counts.max())
    n_f = np.bincount(tri_counts, minlength=fmax + 1).astype(np.float64)
    N1 = np.bincount(f_of_type[c_of_type == 1], minlength=fmax + 1).astype(np.float64)
    Ntypes = np.bincount(f_of_type, minlength=fmax + 1).astype(np.float64)

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["f", "n_contexts", "N1_types", "cont_types", "M", "S_eff"])
        for f in range(1, fmax + 1):
            if n_f[f] > 0:
                w.writerow([f, int(n_f[f]), int(N1[f]), int(Ntypes[f]),
                            round(N1[f] / (f * n_f[f]), 6), round(Ntypes[f] / n_f[f], 4)])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
