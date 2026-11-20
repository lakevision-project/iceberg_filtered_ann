"""Where the index lives, and whether rows are sorted: footer vs Puffin.

    python bench/make_data.py --out data --also-scattered     # once
    python bench/puffin_bench.py                              # -> bench/results/local_1m_puffin.json

Five tables over the SAME 1M x 128 rows (16 files, 250 clusters per file,
nprobe 8, 20 queries, recall@10 against exact search), all searched with the
pyiceberg front end and trace_io on:

  footer-sorted        index in the data file, rows cluster-sorted   (data/indexed)
  footer-unsorted      index in the data file, original row order    (data/indexed_scattered)
  puffin-attach        index ATTACHED to the plain files in place: original order,
                       snappy embeddings -> vectors fetched by row group (data/plain)
  puffin-attach-plain  same, on uncompressed copies of the plain files -> IFVR runs
                       located in the existing bytes, ranged vector reads
  puffin-sorted        rewrite with cluster sort, index in Puffin (write_indexed_table)

Bytes/requests count every read of the data file plus, for Puffin, the index
blob reads. Local timings sit in the page cache; bytes and requests are what
transfer to object storage.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore", message=".*statistics missing.*")

from lakehouse_ann.build import embedding_matrix                    # noqa: E402
from lakehouse_ann.iceberg import STATS, create_table_from_files, vector_search  # noqa: E402
from lakehouse_ann import puffin                                     # noqa: E402
from lakehouse_ann.write import attach_indexes, write_indexed_table  # noqa: E402

FILTERS = {"none": None, "tier = 'p10'": "tier = 'p10'", "tier = 'p1'": "tier = 'p1'"}


def build(data, root, quiet):
    from pyiceberg.catalog.sql import SqlCatalog
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.join(root, "wh"))
    cat = SqlCatalog("bench", uri=f"sqlite:///{root}/cat.db", warehouse=f"file://{root}/wh")
    files = lambda d: [os.path.abspath(p) for p in sorted(glob.glob(os.path.join(data, d, "*.parquet")))]
    t = {}
    t["footer-sorted"] = create_table_from_files(cat, "b.footer_sorted", files("indexed"))
    t["footer-unsorted"] = create_table_from_files(cat, "b.footer_unsorted", files("indexed_scattered"))

    t0 = time.perf_counter()
    tb = create_table_from_files(cat, "b.puffin_attach", files("plain"))
    attach_indexes(tb, "emb", n_clusters=250, log=quiet)
    t["puffin-attach"] = cat.load_table("b.puffin_attach")
    attach_s = time.perf_counter() - t0

    plain_u = os.path.join(root, "plain_uncompressed")
    os.makedirs(plain_u)
    for p in files("plain"):                   # same rows/order, embedding PLAIN + uncompressed
        tab = pq.read_table(p)
        others = [c for c in tab.column_names if c != "emb"]
        pq.write_table(tab, os.path.join(plain_u, os.path.basename(p)), row_group_size=8192,
                       compression={**{c: "zstd" for c in others}, "emb": "none", "emb.list.element": "none"},
                       use_dictionary=others, column_encoding={"emb": "PLAIN"})
    tb = create_table_from_files(cat, "b.puffin_attach_plain",
                                 [os.path.abspath(p) for p in sorted(glob.glob(plain_u + "/*.parquet"))])
    attach_indexes(tb, "emb", n_clusters=250, log=quiet)
    t["puffin-attach-plain"] = cat.load_table("b.puffin_attach_plain")

    t["puffin-sorted"] = write_indexed_table(
        cat, "b.puffin_sorted", files("plain"), "emb", n_clusters=250, row_group_size=8192,
        compress_embedding=False, max_rows_per_file=10**9, index_location="puffin", log=quiet)
    return t, attach_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nprobe", type=int, default=8)
    ap.add_argument("--json", default="bench/results/local_1m_puffin.json")
    a = ap.parse_args()
    root = os.path.join(a.data, "puffin_bench")
    tables, attach_s = build(a.data, root, lambda *x: None)

    truth = pq.read_table(sorted(glob.glob(os.path.join(a.data, "plain", "*.parquet"))))
    vecs, ids = embedding_matrix(truth["emb"]), truth["id"].to_numpy()
    tier = truth["tier"].to_numpy(zero_copy_only=False)
    rng = np.random.default_rng(123)
    qs = vecs[rng.choice(len(vecs), a.queries, replace=False)] + \
        rng.standard_normal((a.queries, vecs.shape[1])).astype(np.float32) * 0.5

    refs_ms = {}
    for name, tbl in tables.items():               # planning cost of the Puffin registry, cold
        puffin._FOOTERS.clear()
        t0 = time.perf_counter()
        n = len(puffin.index_refs(tbl, "emb"))
        refs_ms[name] = ((time.perf_counter() - t0) * 1000, n)

    rows, print_rows = [], []
    print(f"attach (index 16 existing files in place, no rewrite): {attach_s:.1f}s\n")
    print(f"{'filter':14} {'table':20} {'recall':>6} {'MB read':>8} {'reqs':>5} {'idx MB':>6} {'ms':>5}  plans")
    for fname, flt in FILTERS.items():
        mask = np.ones(len(ids), bool) if flt is None else tier == flt.split("'")[1]
        for tname, tbl in tables.items():
            rec, mb, reqs, imb, ms, plans = [], [], [], [], [], {}
            for q in qs:
                d = np.linalg.norm(vecs - q, axis=1)
                d[~mask] = np.inf
                want = set(ids[np.argpartition(d, a.k)[:a.k]].tolist())
                STATS.clear()
                t0 = time.perf_counter()
                out = vector_search(tbl, "emb", q, k=a.k, nprobe=a.nprobe, trace_io=True,
                                    row_filter=flt or "true", selected_fields=("id",))
                ms.append((time.perf_counter() - t0) * 1000)
                rec.append(len(set(out["id"].to_pylist()) & want) / a.k)
                extra = sum(s.index_bytes for s in STATS if s.index_source == "puffin")
                mb.append((sum(s.io_bytes or 0 for s in STATS) + extra) / 1e6)
                reqs.append(sum((s.io_requests or 0) + (s.index_requests if s.index_source == "puffin" else 0)
                                for s in STATS))
                imb.append(sum(s.index_bytes for s in STATS) / 1e6)
                for s in STATS:
                    plans[s.plan] = plans.get(s.plan, 0) + 1 / a.queries
            r = {"filter": fname, "table": tname, "recall": float(np.mean(rec)), "mb": float(np.mean(mb)),
                 "requests": float(np.mean(reqs)), "index_mb": float(np.mean(imb)),
                 "ms": float(np.median(ms)), "plans": plans}
            rows.append(r)
            ps = ", ".join(f"{p}×{c:g}" for p, c in sorted(plans.items()))
            print(f"{fname:14} {tname:20} {r['recall']:6.2f} {r['mb']:8.1f} {r['requests']:5.0f} "
                  f"{r['index_mb']:6.1f} {r['ms']:5.0f}  {ps}", flush=True)
    print("\nregistry planning cost (cold, footer reads):",
          {k: f"{v[0]:.1f} ms / {v[1]} refs" for k, v in refs_ms.items() if v[1]})
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    json.dump({"rows": rows, "attach_s": attach_s, "registry_ms": refs_ms,
               "queries": a.queries, "k": a.k, "nprobe": a.nprobe}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
