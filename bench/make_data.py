"""Synthetic benchmark table, written once plain and once footer-indexed.

Vectors are a Gaussian mixture (topics) so IVF has real structure to exploit.
Rows are scattered across files at random, so file-level pruning gets no help:
every file holds a little of every topic and every filter value. That is the
worst case for the per-file design and the honest one to show.

Columns
  id       int64
  tier     'p50' / 'p10' / 'p1' / 'p01'   ≈ 50/10/1/0.1 % — independent of the vector
  region   'r0'..'r19'                    a function of the topic — CORRELATED with the vector
  price    float64 uniform [0, 100)
  emb      FixedSizeList<float32, dim>
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lakehouse_ann.build import index_table  # noqa: E402

TIERS = np.array(["p50", "p10", "p1", "p01"])
TIER_P = np.array([0.50, 0.10, 0.01, 0.001])


def make(n_rows, dim, n_topics, seed):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_topics, dim)).astype(np.float32) * 3
    topic = rng.integers(0, n_topics, n_rows)
    emb = centers[topic] + rng.standard_normal((n_rows, dim)).astype(np.float32)
    p = np.append(TIER_P, 1 - TIER_P.sum())
    tier = np.append(TIERS, "rest")[rng.choice(5, n_rows, p=p)]
    return pa.table({
        "id": np.arange(n_rows, dtype=np.int64),
        "tier": tier,
        "region": np.char.add("r", (topic % 20).astype(str)),
        "price": rng.uniform(0, 100, n_rows),
        "emb": pa.FixedSizeListArray.from_arrays(pa.array(emb.ravel()), dim),
    }), rng.permutation(n_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--files", type=int, default=16)
    ap.add_argument("--topics", type=int, default=2000)
    ap.add_argument("--n-clusters", type=int, default=0, help="per file; default sqrt(rows/file)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--also-scattered", action="store_true",
                    help="also write indexed_scattered/ (original row order kept)")
    a = ap.parse_args()

    table, perm = make(a.rows, a.dim, a.topics, a.seed)
    for sub in ("plain", "indexed", "indexed_scattered"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    per = a.rows // a.files
    for f in range(a.files):
        t = table.take(pa.array(np.sort(perm[f * per:(f + 1) * per])))
        name = f"part-{f:03d}.parquet"
        pq.write_table(t, os.path.join(a.out, "plain", name), row_group_size=8192)
        t0 = time.perf_counter()
        k = a.n_clusters or int(np.sqrt(per))
        n = index_table(t, os.path.join(a.out, "indexed", name), "emb", k)
        if a.also_scattered:
            index_table(t, os.path.join(a.out, "indexed_scattered", name), "emb", k, cluster_rows=False)
        print(f"{name}: {per} rows, {k} clusters, {n / 1e6:.1f} MB, built in {time.perf_counter() - t0:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
