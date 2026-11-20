"""Filtered vector search with only pyiceberg, on a local SQLite catalog.

Two tables over the same 1M rows (bench/make_data.py output):

  vec.docs            16 indexed files, rows spread at random (no layout)
  vec.docs_by_region  20 indexed files, one per region, partitioned
                      identity(region) — the "pure files" layout

Then: recall / files planned / measured I/O per filter on both tables, a
copy-on-write delete (pyiceberg rewrites the file, the rewrite has no index,
search stays correct by scanning it exactly), and time travel back to the
indexed snapshot.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import time
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pyiceberg.catalog.sql import SqlCatalog  # noqa: E402

from lakehouse_ann.build import embedding_matrix, index_table  # noqa: E402
from lakehouse_ann.iceberg import STATS, create_table_from_files, vector_search  # noqa: E402

warnings.filterwarnings("ignore", message=".*statistics missing.*")

FILTERS = {
    "none": (None, lambda t: None),
    "tier = 'p10'": ("tier = 'p10'", lambda t: pc.equal(t["tier"], "p10")),
    "tier = 'p1'": ("tier = 'p1'", lambda t: pc.equal(t["tier"], "p1")),
    "region = 'r3'": ("region = 'r3'", lambda t: pc.equal(t["region"], "r3")),
    "region = 'r3' and tier = 'p10'": ("region = 'r3' and tier = 'p10'",
                                       lambda t: pc.and_(pc.equal(t["region"], "r3"), pc.equal(t["tier"], "p10"))),
}


def by_region_files(src_dir: str, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(out_dir, "*.parquet")))
    if existing:
        return existing
    t = pq.read_table(sorted(glob.glob(os.path.join(src_dir, "*.parquet"))))
    t = t.replace_schema_metadata(None)
    for region in sorted(set(t["region"].to_pylist())):
        part = t.filter(pc.equal(t["region"], region))
        index_table(part, os.path.join(out_dir, f"{region}.parquet"), "emb", int(np.sqrt(part.num_rows)))
        print(f"  indexed {region}: {part.num_rows} rows", flush=True)
    return sorted(glob.glob(os.path.join(out_dir, "*.parquet")))


def recall(got_ids, truth_tbl, vecs, q, k, pred):
    ids = truth_tbl["id"].to_numpy()
    m = np.ones(len(ids), bool) if pred is None else pred.to_numpy(zero_copy_only=False)
    want = ids[m][np.argsort(np.linalg.norm(vecs[m] - q, axis=1))[:k]]
    return len(set(want.tolist()) & set(got_ids)) / max(len(want), 1)


def run(tbl, q, k, row_filter, **kw):
    n0 = len(STATS)
    t0 = time.perf_counter()
    out = vector_search(tbl, "emb", q, k, row_filter=row_filter or "true",
                        selected_fields=("id", "tier", "region"), trace_io=True, **kw)
    ms = (time.perf_counter() - t0) * 1000
    return out, ms, STATS[n0:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()

    root = os.path.abspath(os.path.join(a.data, "iceberg"))
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.join(root, "warehouse"))
    cat = SqlCatalog("demo", uri=f"sqlite:///{root}/catalog.db", warehouse=f"file://{root}/warehouse")
    cat.create_namespace("vec")

    docs_files = [os.path.abspath(p) for p in sorted(glob.glob(os.path.join(a.data, "indexed", "*.parquet")))]
    print("building region-pure files (once)...")
    region_files = [os.path.abspath(p) for p in
                    by_region_files(os.path.join(a.data, "indexed"), os.path.join(a.data, "by_region"))]
    docs = create_table_from_files(cat, "vec.docs", docs_files)
    by_region = create_table_from_files(cat, "vec.docs_by_region", region_files, partition_by=["region"])
    print(f"vec.docs: {len(docs_files)} files | vec.docs_by_region: {len(region_files)} files, "
          f"spec {by_region.spec()}\n")

    truth_tbl = pq.read_table(docs_files)
    vecs = embedding_matrix(truth_tbl["emb"])
    rng = np.random.default_rng(123)
    queries = vecs[rng.choice(len(vecs), a.queries, replace=False)] + \
        rng.standard_normal((a.queries, vecs.shape[1])).astype(np.float32) * 0.5

    print(f"{'filter':<32} {'table':<18} {'recall':>6} {'files':>5} {'ms':>6} {'MB read':>8} {'reqs':>5}  plans")
    for name, (row_filter, pred_fn) in FILTERS.items():
        pred = pred_fn(truth_tbl)
        for label, tbl in (("docs", docs), ("docs_by_region", by_region)):
            rs, mss, mbs, reqs, files, plans = [], [], [], [], [], {}
            for q in queries:
                out, ms, st = run(tbl, q, a.k, row_filter)
                rs.append(recall(out["id"].to_pylist() if out.num_rows else [], truth_tbl, vecs, q, a.k, pred))
                mss.append(ms)
                mbs.append(sum(s.io_bytes or 0 for s in st) / 1e6)
                reqs.append(sum(s.io_requests or 0 for s in st))
                files.append(len(st))
                for s in st:
                    plans[s.plan] = plans.get(s.plan, 0) + 1 / a.queries
            plan_s = ", ".join(f"{p}×{c:g}" for p, c in sorted(plans.items()))
            print(f"{name:<32} {label:<18} {np.mean(rs):6.2f} {np.mean(files):5.0f} {np.median(mss):6.0f} "
                  f"{np.mean(mbs):8.1f} {np.mean(reqs):5.0f}  {plan_s}", flush=True)

    # ── copy-on-write delete + time travel ───────────────────────────────────
    print("\nDELETE FROM vec.docs_by_region WHERE region = 'r3' AND tier = 'p50'  (pyiceberg: copy-on-write)")
    before = by_region.current_snapshot().snapshot_id
    by_region.delete("region = 'r3' and tier = 'p50'")
    by_region = cat.load_table("vec.docs_by_region")
    q = queries[0]
    out, ms, st = run(by_region, q, a.k, "region = 'r3'")
    still = pc.and_(pc.equal(truth_tbl["region"], "r3"), pc.not_equal(truth_tbl["tier"], "p50"))
    print(f"  after delete : recall {recall(out['id'].to_pylist(), truth_tbl, vecs, q, a.k, still):.2f} "
          f"vs post-delete truth, deleted tier present: {'p50' in out['tier'].to_pylist()}, "
          f"plans {[s.plan for s in st]}  ({ms:.0f} ms)")
    out, ms, st = run(by_region, q, a.k, "region = 'r3'", snapshot_id=before)
    print(f"  time travel  : recall {recall(out['id'].to_pylist(), truth_tbl, vecs, q, a.k, pc.equal(truth_tbl['region'], 'r3')):.2f} "
          f"vs pre-delete truth, plans {[s.plan for s in st]}  ({ms:.0f} ms)")


if __name__ == "__main__":
    main()
