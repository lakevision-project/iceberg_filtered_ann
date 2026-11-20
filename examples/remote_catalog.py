"""Filtered vector search against YOUR Iceberg catalog (REST, Glue, Hive, SQL, ...).

The catalog comes from pyiceberg's standard configuration — nothing about it
is hard-coded here. For example, a REST catalog over S3-compatible storage:

    export PYICEBERG_CATALOG__PROD__TYPE=rest
    export PYICEBERG_CATALOG__PROD__URI=https://iceberg-rest.example.com
    export PYICEBERG_CATALOG__PROD__S3__ENDPOINT=https://s3.example.com
    export PYICEBERG_CATALOG__PROD__S3__ACCESS_KEY_ID=...
    export PYICEBERG_CATALOG__PROD__S3__SECRET_ACCESS_KEY=...
    export PYICEBERG_CATALOG__PROD__S3__FORCE_VIRTUAL_ADDRESSING=false   # path-style stores

then:

    python examples/remote_catalog.py --catalog prod --table vec.docs \\
        --query "how do I reset a lost password" \\
        --where "category = 'tech'" --where "category = 'tech' AND customer_id < 200" --gt

Each query runs through the SQL front end (Daft executes; add --ray to use a
Ray cluster) or, with --frontend pyiceberg, without any query engine. It
prints the hits and, per query, the files searched, plans and bytes read.
--gt adds exact recall@k for the last --where (reads that filter's rows in full).

The query text is embedded with the table's model (sentence-transformers,
--model); pass --query-npy to supply a vector instead.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore", message=".*statistics missing.*")

from lakehouse_ann.search import STATS  # noqa: E402


def report(label, out, ms, text_col):
    st = STATS.copy()
    plans = {}
    for s in st:
        plans[s.plan] = plans.get(s.plan, 0) + 1
    vmb = sum(s.vector_bytes_read for s in st) / 1e6
    imb = sum(s.index_bytes for s in st) / 1e6
    print(f"\n── {label} ──")
    print(f"   files searched {len(st)} | plans {plans} | index {imb:.1f} MB, "
          f"vectors {vmb:.1f} MB read | {ms / 1000:.2f} s"
          + ("" if st else "  (per-file stats stay on Ray workers)"))
    texts = out.get(text_col)
    for i in range(len(out["_distance"])):
        text = str(texts[i] or "")[:72].replace("\n", " ") if texts else ""
        print(f"   {out['_distance'][i]:8.4f}  {text}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="default")
    ap.add_argument("--table", required=True)
    ap.add_argument("--column", default="embedding")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--text-col", default="text", help="column printed with each hit")
    q = ap.add_mutually_exclusive_group(required=True)
    q.add_argument("--query")
    q.add_argument("--query-npy")
    ap.add_argument("--model", default="ibm-granite/granite-embedding-97m-multilingual-r2")
    ap.add_argument("--where", action="append", default=[])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--snapshot-id", type=int)
    ap.add_argument("--frontend", choices=("sql", "pyiceberg"), default="sql")
    ap.add_argument("--ray", nargs="?", const="", default=None, metavar="ADDRESS")
    ap.add_argument("--gt", action="store_true")
    a = ap.parse_args()

    import daft
    if a.ray is not None:
        daft.set_runner_ray(address=a.ray or None)
    from pyiceberg.catalog import load_catalog
    tbl = load_catalog(a.catalog).load_table(a.table)
    names = tbl.schema().column_names
    cols = [c for c in (a.id_col, a.text_col) if c in names]
    files = list(tbl.scan(snapshot_id=a.snapshot_id).plan_files())
    print(f"{a.table} @ snapshot {a.snapshot_id or tbl.metadata.current_snapshot_id}: "
          f"{len(files)} files, {sum(f.file.record_count for f in files):,} rows, "
          f"{sum(f.file.file_size_in_bytes for f in files) / 1e9:.2f} GB, spec {tbl.spec()}")

    if a.query_npy:
        vec = np.load(a.query_npy).astype(np.float32).reshape(-1)
    else:
        from lakehouse_ann.embed import embed_text
        t0 = time.perf_counter()
        vec = np.asarray(embed_text(daft.from_pydict({"t": [a.query]}), "t", model=a.model)
                         .to_pydict()["embedding"][0], np.float32)
        print(f"embedded {a.query!r} with {a.model} ({time.perf_counter() - t0:.1f}s)")

    alias = a.table.split(".")[-1]
    out = {}
    for where in [""] + a.where:
        STATS.clear()
        t0 = time.perf_counter()
        if a.frontend == "sql":
            from lakehouse_ann.iceberg_sql import ann_sql
            sql = (f"SELECT {', '.join(cols + ['_distance'])} FROM {alias} "
                   f"{'WHERE ' + where if where else ''} ORDER BY _distance LIMIT {a.k}")
            out = ann_sql(tbl, a.column, vec, sql, k=a.k, nprobe=a.nprobe,
                          snapshot_id=a.snapshot_id).to_pydict()
        else:
            from lakehouse_ann.iceberg import vector_search
            from pyiceberg.expressions import AlwaysTrue
            out = vector_search(tbl, a.column, vec, k=a.k, nprobe=a.nprobe,
                                row_filter=where or AlwaysTrue(), selected_fields=tuple(cols),
                                snapshot_id=a.snapshot_id).to_pydict()
        report(where or "no filter", out, (time.perf_counter() - t0) * 1000, a.text_col)

    if a.gt and a.id_col in names:
        from lakehouse_ann.build import embedding_matrix
        where = a.where[-1] if a.where else None
        t0 = time.perf_counter()
        kw = {"row_filter": where} if where else {}
        t = tbl.scan(selected_fields=(a.id_col, a.column), snapshot_id=a.snapshot_id, **kw).to_arrow()
        d = np.linalg.norm(embedding_matrix(t[a.column]) - vec, axis=1)
        want = set(t[a.id_col].to_numpy()[np.argsort(d, kind="stable")[: a.k]].tolist())
        got = set(out.get(a.id_col, []))
        print(f"\n── exact top-{a.k} for {where or 'no filter'} ({t.num_rows:,} rows read in "
              f"{time.perf_counter() - t0:.0f}s): recall@{a.k} = {len(want & got)}/{len(want)}")


if __name__ == "__main__":
    main()
