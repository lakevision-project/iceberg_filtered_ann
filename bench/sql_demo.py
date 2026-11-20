"""Validate the SQL front end: SQL over Iceberg-planned, Daft-executed filtered ANN.

Uses the same SQLite catalog and tables as bench/iceberg_demo.py:

  vec.docs            16 indexed files, rows spread at random
  vec.docs_by_region  20 files, identity(region); the r3 file was rewritten
                      by an earlier copy-on-write delete, so the current
                      snapshot mixes 19 indexed files + 1 plain one, and the
                      first snapshot is all-indexed (time travel)

Run from the repo root (build the catalog first with bench/iceberg_demo.py):

    .venv/bin/python bench/sql_demo.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore", message=".*statistics missing.*")

import daft  # noqa: E402
from pyiceberg.catalog.sql import SqlCatalog  # noqa: E402

from lakehouse_ann.build import embedding_matrix  # noqa: E402
from lakehouse_ann.iceberg import STATS, vector_search  # noqa: E402
from lakehouse_ann.iceberg_sql import ann_sql, read_ivf_iceberg  # noqa: E402

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "iceberg")
CATALOG = f"sqlite:///{_ROOT}/catalog.db"
WAREHOUSE = f"file://{_ROOT}/warehouse"


def truth(tbl, row_filter, q, k, snapshot_id=None):
    """Exact top-k ids among matching rows, straight through pyiceberg."""
    scan = tbl.scan(snapshot_id=snapshot_id)
    if row_filter:
        scan = tbl.scan(row_filter=row_filter, snapshot_id=snapshot_id)
    t = scan.to_arrow()
    emb = embedding_matrix(t.column("emb"))
    d = np.linalg.norm(emb - q, axis=1)
    order = np.argsort(d, kind="stable")[:k]
    return t.column("id").to_pylist(), order, d


def recall_of(result_ids, tids, order):
    want = {tids[i] for i in order}
    return len(want & set(result_ids)) / len(want)


def run_sql(tbl, q, sql, k=10, **kw):
    STATS.clear()
    t0 = time.perf_counter()
    df = ann_sql(tbl, "emb", q, sql, k=k, **kw)
    out = df.to_pydict()
    ms = (time.perf_counter() - t0) * 1000
    st = STATS.copy()
    return out, st, ms


def main():
    cat = SqlCatalog("demo", uri=CATALOG, warehouse=WAREHOUSE)
    docs = cat.load_table("vec.docs")
    by_region = cat.load_table("vec.docs_by_region")

    first = next(iter(docs.scan().plan_files())).file.file_path
    rng = np.random.default_rng(123)
    q = embedding_matrix(pq.read_table(first, columns=["emb"]).column("emb"))[0] \
        + rng.standard_normal(128).astype(np.float32) * 0.5

    print("── vec.docs: SQL filters, recall vs exact pyiceberg ground truth ──")
    print(f"{'SQL WHERE':38} {'rows':>4} {'recall':>6} {'files':>5} {'plans':>22} {'ms':>6}")
    cases = [
        ("tier = 'p10'", "tier = 'p10'"),
        ("region = 'r3'  (correlated)", "region = 'r3'"),
        ("region = 'r3' AND tier = 'p10'", "region = 'r3' and tier = 'p10'"),
    ]
    for label, flt in cases:
        sql = (f"SELECT id, tier, region, _distance FROM docs WHERE {flt} "
               "ORDER BY _distance LIMIT 10")
        out, st, ms = run_sql(docs, q, sql)
        tids, order, _ = truth(docs, flt, q, 10)
        plans = ",".join(sorted({s.plan for s in st}))
        print(f"{label:38} {len(out['id']):>4} {recall_of(out['id'], tids, order):>6.2f} "
              f"{len(st):>5} {plans:>22} {ms:>6.0f}")

    print("\n── vec.docs_by_region: the same SQL, now with Iceberg pruning ──")
    sql = ("SELECT id, region, tier, _distance FROM docs_by_region WHERE region = 'r3' "
           "ORDER BY _distance LIMIT 10")
    out, st, ms = run_sql(by_region, q, sql)
    tids, order, _ = truth(by_region, "region = 'r3'", q, 10)   # post-delete truth
    plans = ",".join(sorted({s.plan for s in st}))
    print(f"region = 'r3'  (current snapshot): rows {len(out['id'])}, recall "
          f"{recall_of(out['id'], tids, order):.2f}, files searched {len(st)} "
          f"({plans}), {ms:.0f} ms")
    print("   ^ 19 of 20 files pruned by partition pruning; the survivor is the")
    print("     copy-on-write rewrite (no index) -> exact scan, answer still right")

    snap0 = by_region.metadata.snapshots[0].snapshot_id
    sql = ("SELECT id, region, _distance FROM docs_by_region WHERE region = 'r3' "
           "ORDER BY _distance LIMIT 10")
    out, st, ms = run_sql(by_region, q, sql, snapshot_id=snap0)
    tids, order, _ = truth(by_region, "region = 'r3'", q, 10, snapshot_id=snap0)
    print(f"region = 'r3'  (snapshot {str(snap0)[-6:]}): rows {len(out['id'])}, recall "
          f"{recall_of(out['id'], tids, order):.2f}, files searched {len(st)} "
          f"({','.join(sorted({s.plan for s in st}))}), {ms:.0f} ms  <- time travel, indexed")

    print("\n── SQL composition: JOIN and aggregation over the ANN result ──")
    meta = daft.from_pydict({"region": ["r3", "r7", "r11"], "label": ["eu", "us", "apac"]})
    sql = ("SELECT a.id, m.label, a._distance FROM docs a JOIN meta m ON a.region = m.region "
           "WHERE a.tier = 'p1' ORDER BY a._distance LIMIT 10")
    out, st, ms = run_sql(docs, q, sql, bindings={"meta": meta})
    print(f"JOIN: rows {len(out['id'])}, labels {out['label']}, "
          f"filter pushed into kernel: {any(s.n_match is not None for s in st)}")

    sql = ("SELECT region, count(*) AS n FROM (SELECT region, _distance FROM docs "
           "WHERE tier = 'p1' ORDER BY _distance LIMIT 10) GROUP BY region")
    out, _, ms = run_sql(docs, q, sql)
    print(f"GROUP BY over top-k subquery: {dict(zip(out['region'], out['n']))} ({ms:.0f} ms)")

    print("\n── guards ──")
    sql = ("SELECT id FROM (SELECT id, tier, _distance FROM docs ORDER BY _distance LIMIT 10) "
           "WHERE tier = 'p1'")
    out, _, _ = run_sql(docs, q, sql)
    print(f"post-filter trap (WHERE after ORDER BY..LIMIT): {len(out['id'])} rows "
          "(unavoidable: the filter cannot be pushed through the limit)")

    sql = "SELECT id, _distance FROM docs WHERE tier = 'nope' ORDER BY _distance LIMIT 10"
    out, st, _ = run_sql(docs, q, sql)
    print(f"no-match filter: {len(out['id'])} rows, all files skipped: "
          f"{all(s.plan == 'skip:no-match' for s in st)}")

    sql = "SELECT id, _distance FROM docs WHERE _distance < 1e9 ORDER BY _distance LIMIT 10"
    out, _, _ = run_sql(docs, q, sql)
    print(f"WHERE on _distance (always true): {len(out['id'])} rows, applied after ranking")

    out_lim, _, _ = run_sql(docs, q, "SELECT id FROM docs WHERE tier = 'p10' "
                             "ORDER BY _distance LIMIT 15")
    tids, order, _ = truth(docs, "tier = 'p10'", q, 15)
    print(f"LIMIT 15 > registered k=10: {len(out_lim['id'])} rows, recall "
          f"{recall_of(out_lim['id'], tids, order):.2f} (k auto-raised by ann_sql)")

    print("\n── parity with the pyiceberg front end (same kernel, same plan) ──")
    STATS.clear()
    ref = vector_search(docs, "emb", q, k=10, row_filter="tier = 'p10'",
                        selected_fields=("id",))
    ref_ids = ref.column("id").to_pylist()
    out, st, _ = run_sql(docs, q, "SELECT id FROM docs WHERE tier = 'p10' "
                         "ORDER BY _distance LIMIT 10")
    same = out["id"] == ref_ids
    print(f"identical ids & order vs vector_search(): {same} "
          f"({len(ref_ids)} vs {len(out['id'])} rows)")

    df = read_ivf_iceberg(docs, "emb", q, k=10).where(daft.col("tier") == "p10") \
        .sort("_distance").limit(10)
    print(f"DataFrame API (read_ivf_iceberg + where/sort/limit): "
          f"{df.to_pydict()['id'] == ref_ids}")


if __name__ == "__main__":
    main()
