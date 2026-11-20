"""Filtered top-k on Daft: brute force vs footer-index search vs post-filtering.

  brute    daft.read_parquet + where + euclidean_distance + sort + limit
           (exact; reads every vector)
  ivf      ivf_search(..., where=f)  — filter pushed into each per-file task,
           applied DURING ranking
  post     read_ivf().sort().limit(k).where(f) — the filter lands AFTER the
           top-k, which is what you get from an index that cannot see filters

Recall@k is against exact numpy ground truth. All three read the SAME files:
the indexed files are ordinary Parquet to daft.read_parquet.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import daft  # noqa: E402
from daft import col, lit  # noqa: E402
from daft.functions import euclidean_distance  # noqa: E402

from lakehouse_ann import STATS, ivf_search, read_ivf  # noqa: E402
from lakehouse_ann.build import embedding_matrix  # noqa: E402
from lakehouse_ann.search import column_bytes  # noqa: E402

# Object-storage model (labelled as a MODEL in every output): one GET costs a
# round trip, transfers share one node's bandwidth, GETs run CONC at a time.
S3_RTT_MS, S3_GBPS, S3_CONC = 30.0, 1.0, 64


def s3_model_ms(nbytes, requests):
    return np.ceil(requests / S3_CONC) * S3_RTT_MS + nbytes / (S3_GBPS * 1e9) * 1000

FILTERS = {
    "none":            (None, None),
    "tier=p50 (50%)":  (col("tier") == "p50", lambda t: pc.equal(t["tier"], "p50")),
    "tier=p10 (10%)":  (col("tier") == "p10", lambda t: pc.equal(t["tier"], "p10")),
    "tier=p1 (1%)":    (col("tier") == "p1", lambda t: pc.equal(t["tier"], "p1")),
    "tier=p01 (0.1%)": (col("tier") == "p01", lambda t: pc.equal(t["tier"], "p01")),
    "region=r3 (5%, correlated)": (col("region") == "r3", lambda t: pc.equal(t["region"], "r3")),
    "price<5 & tier=p10 (0.5%)": ((col("price") < 5) & (col("tier") == "p10"),
                                  lambda t: pc.and_(pc.less(t["price"], 5), pc.equal(t["tier"], "p10"))),
}


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/indexed")
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nprobe", type=int, default=8)
    ap.add_argument("--probe-rule", default="matched", choices=["matched", "all"])
    ap.add_argument("--overfetch", type=int, default=16)
    ap.add_argument("--only", default="", help="comma-separated substrings of filter names to run")
    ap.add_argument("--json", default="bench/results/local_1m.json")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.data, "*.parquet")))
    truth_tbl = pq.read_table(files)
    ids = truth_tbl["id"].to_numpy()
    vecs = embedding_matrix(truth_tbl["emb"])
    dim = vecs.shape[1]
    emb_bytes = vecs.nbytes
    rng = np.random.default_rng(123)
    queries = vecs[rng.choice(len(vecs), a.queries, replace=False)] + \
        rng.standard_normal((a.queries, dim)).astype(np.float32) * 0.5
    qtype = daft.DataType.fixed_size_list(daft.DataType.float32(), dim)
    pattern = os.path.join(a.data, "*.parquet")
    print(f"{len(ids):,} rows x {dim} dims in {len(files)} files; "
          f"{a.queries} queries, k={a.k}, nprobe={a.nprobe}\n")

    results = []
    for name, (expr, arrow_pred) in FILTERS.items():
        if a.only and not any(o in name for o in a.only.split(",")):
            continue
        mask = np.ones(len(ids), bool) if arrow_pred is None else \
            arrow_pred(truth_tbl).to_numpy(zero_copy_only=False)
        sel = mask.mean()
        row = {"filter": name, "selectivity": float(sel)}
        acc = {m: {"ms": [], "recall": [], "rows": []} for m in ("brute", "ivf", "post")}
        stats = []
        for q in queries:
            d = np.linalg.norm(vecs[mask] - q, axis=1)
            want = set(ids[mask][np.argsort(d)[:a.k]].tolist())

            def brute():
                df = daft.read_parquet(pattern)
                if expr is not None:
                    df = df.where(expr)
                return df.with_column("_distance", euclidean_distance(col("emb"), lit(q.tolist()).cast(qtype))) \
                    .sort("_distance").limit(a.k).select("id").to_pydict()["id"]

            def ivf():
                return ivf_search(pattern, "emb", q, a.k, where=expr, select=["id"],
                                  nprobe=a.nprobe, probe_rule=a.probe_rule, filtered_overfetch=a.overfetch,
                                  trace_io=True).to_pydict()["id"]

            def post():
                df = read_ivf(pattern, "emb", q, a.k, nprobe=a.nprobe).sort("_distance").limit(a.k)
                if expr is not None:
                    df = df.where(expr)
                return df.select("id").to_pydict()["id"]

            for m, fn in (("brute", brute), ("ivf", ivf), ("post", post)):
                n0 = len(STATS)
                got, ms = timed(fn)
                if m == "ivf":
                    stats.extend(STATS[n0:])
                acc[m]["ms"].append(ms)
                acc[m]["rows"].append(len(got))
                acc[m]["recall"].append(len(want & set(got)) / max(len(want), 1))

        for m, v in acc.items():
            row[m] = {"p50_ms": float(np.median(v["ms"])), "recall": float(np.mean(v["recall"])),
                      "avg_rows": float(np.mean(v["rows"]))}
        nq = a.queries
        # Brute force reads at least these column chunks (id, filter columns,
        # embedding) from every file: one GET per chunk at best.
        cols = ["id", "emb"] + ([] if expr is None else sorted(
            {c for c in ("tier", "region", "price") if c in str(expr)}))
        b_bytes = sum(column_bytes(pq.read_metadata(f), cols) for f in files)
        b_reqs = sum(pq.read_metadata(f).num_row_groups * len(cols) + 1 for f in files)
        i_bytes = sum(s.io_bytes or 0 for s in stats) / nq
        i_reqs = sum(s.io_requests or 0 for s in stats) / nq
        row["io"] = {"brute_MB": b_bytes / 1e6, "brute_requests": b_reqs,
                     "ivf_MB": i_bytes / 1e6, "ivf_requests": i_reqs,
                     "s3_model_brute_ms": float(s3_model_ms(b_bytes, b_reqs)),
                     "s3_model_ivf_ms": float(s3_model_ms(i_bytes, i_reqs))}
        plans = {}
        for s in stats:
            plans[s.plan] = plans.get(s.plan, 0) + 1
        row["ivf_trace"] = {
            "vector_MB_per_query": sum(s.vector_bytes_read for s in stats) / nq / 1e6,
            "vector_requests_per_query": sum(s.vector_requests for s in stats) / nq,
            "candidates_per_query": sum(s.candidates for s in stats) / nq,
            "clusters_probed_per_query": sum(s.clusters_probed for s in stats) / nq,
            "plans_per_query": {p: c / nq for p, c in sorted(plans.items())},
        }
        results.append(row)
        print(f"{name:<28} sel={sel:7.3%}  "
              f"brute {row['brute']['p50_ms']:6.0f}ms r={row['brute']['recall']:.2f} | "
              f"ivf {row['ivf']['p50_ms']:5.0f}ms r={row['ivf']['recall']:.2f} "
              f"({row['ivf_trace']['vector_MB_per_query']:.1f}MB/{row['ivf_trace']['vector_requests_per_query']:.0f}req, "
              f"{row['ivf_trace']['clusters_probed_per_query']:.0f}cl) | "
              f"post r={row['post']['recall']:.2f} rows={row['post']['avg_rows']:.1f}  "
              f"plans={row['ivf_trace']['plans_per_query']}", flush=True)
        io = row["io"]
        print(f"{'':<28} I/O brute {io['brute_MB']:6.1f}MB/{io['brute_requests']}req  "
              f"ivf {io['ivf_MB']:6.1f}MB/{io['ivf_requests']:.0f}req  |  S3 MODEL "
              f"brute {io['s3_model_brute_ms']:.0f}ms  ivf {io['s3_model_ivf_ms']:.0f}ms", flush=True)

    with open(a.json, "w") as f:
        json.dump({"rows": len(ids), "dim": dim, "files": len(files), "k": a.k, "nprobe": a.nprobe,
                   "queries": a.queries,
                   "s3_model": {"rtt_ms": S3_RTT_MS, "GBps": S3_GBPS, "concurrency": S3_CONC},
                   "probe_rule": a.probe_rule, "data": a.data, "embedding_MB": emb_bytes / 1e6, "results": results}, f, indent=1)
    print(f"\nembedding column total: {emb_bytes / 1e6:.0f} MB; results -> {a.json}")


if __name__ == "__main__":
    main()
