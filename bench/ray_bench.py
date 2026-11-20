"""Filtered ANN over Iceberg on Daft's Ray runner: timing, recall, A/B.

    python bench/ray_bench.py --catalog prod --table vec.docs \\
        --where "band=customer_id >= 400 AND customer_id < 600" \\
        --where "rare=customer_id IN (7, 207, 407)" \\
        --queries queries.npy --gt

  * every --table is queried with every --where case (plus no filter), k / nprobe
    as given, once per query vector, after one warm-up query;
  * --runner ray (default) | native | both — "both" runs native and Ray in
    separate processes (Daft's runner is fixed per process) and compares top-k;
  * --gt computes exact ground truth with one Ray task per data file of the
    FIRST table (the filter is the same WHERE, evaluated with pyarrow), cached
    in --gt-cache, so recall@k is reported per query;
  * --filter-reads per_rg batched  A/Bs SearchConfig.filter_read (any other
    SearchConfig field can be set with --cfg key=value).

Catalogs come from pyiceberg's config (~/.pyiceberg.yaml or PYICEBERG_CATALOG__*).
Query vectors: a .npy of shape (n, dim) — embed them with lakehouse_ann.embed.
Per-file I/O stats stay in the worker processes under Ray, so this reports
wall time, recall, files planned and where the work ran.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def parse_cases(wheres):
    cases = [("no-filter", "")]
    for w in wheres or []:
        head, sep, rest = w.partition("=")
        if sep and head.strip().isidentifier() and not rest.startswith("="):
            cases.append((head.strip(), rest.strip()))
        else:
            cases.append((w.strip()[:24], w.strip()))
    return cases


# ── exact ground truth (Ray tasks over the first table's files) ─────────────

def ground_truth(tbl, column, id_col, queries, cases, k, cache):
    if cache and os.path.exists(cache):
        raw = json.load(open(cache))
        return {tuple(json.loads(key)): set(v) for key, v in raw.items()}
    import ray
    from pyiceberg.expressions.parser import parse
    from lakehouse_ann.iceberg import _arrow_filter

    filters = []
    for name, cond in cases:
        expr, cols = _arrow_filter(tbl, parse(cond)) if cond else (None, [])
        filters.append((name, expr, list(cols)))

    @ray.remote(num_cpus=2, memory=4 * 1024**3, max_retries=2)
    def one(path, io, qs):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from lakehouse_ann.build import embedding_matrix
        need = {id_col, column} | {c for _, _, cs in filters for c in cs}
        with io.new_input(path).open() as f:
            t = pq.ParquetFile(f).read(columns=sorted(need))
        E = embedding_matrix(t[column])
        d2 = (E * E).sum(1)[None, :] - 2 * qs @ E.T + (qs * qs).sum(1)[:, None]
        ids = t[id_col].to_numpy()
        pos = pa.array(np.arange(t.num_rows))
        out = {}
        for name, expr, _ in filters:
            m = np.ones(t.num_rows, bool)
            if expr is not None:
                keep = t.append_column("__pos", pos).filter(expr)["__pos"].to_numpy()
                m = np.zeros(t.num_rows, bool)
                m[keep] = True
            for qi in range(len(qs)):
                dd = np.where(m, d2[qi], np.inf)
                top = np.argpartition(dd, min(k, len(dd) - 1))[:k]
                top = top[np.isfinite(dd[top])]
                out[(name, qi)] = list(zip(dd[top].tolist(), ids[top].tolist()))
        return out

    t0 = time.perf_counter()
    files = [t.file.file_path for t in tbl.scan().plan_files()]
    qref = ray.put(queries)
    merged = {}
    for part in ray.get([one.remote(p, tbl.io, qref) for p in files]):
        for key, lst in part.items():
            merged.setdefault(key, []).extend(lst)
    gt = {key: [i for _, i in sorted(v)[:k]] for key, v in merged.items()}
    print(f"ground truth: {len(files)} files, {time.perf_counter() - t0:.0f}s", flush=True)
    if cache:
        json.dump({json.dumps(list(key)): v for key, v in gt.items()}, open(cache, "w"))
    return {key: set(v) for key, v in gt.items()}


# ── one benchmark pass (one process, one runner) ───────────────────────────

def run_pass(a, runner):
    import daft
    if runner == "ray":
        daft.set_runner_ray(address=a.address)
    else:
        daft.set_runner_native()
    from pyiceberg.catalog import load_catalog
    from lakehouse_ann.iceberg_sql import ann_sql

    cat = load_catalog(a.catalog)
    queries = np.load(a.queries).astype(np.float32)
    queries = queries.reshape(-1, queries.shape[-1])[: a.n_queries]
    cases = parse_cases(a.where)
    cfg_base = {}
    for kv in a.cfg:
        key, val = kv.split("=", 1)
        cfg_base[key] = int(val) if val.lstrip("-").isdigit() else val
    tables = [cat.load_table(t) for t in a.table]

    gt = None
    if a.gt:
        import ray
        ray.init(address=a.address or ("auto" if runner == "ray" else None), ignore_reinit_error=True)
        gt = ground_truth(tables[0], a.column, a.id_col, queries, cases, a.k, a.gt_cache)

    rows = []
    for tname, tbl in zip(a.table, tables):
        alias = tname.split(".")[-1]
        for fr in a.filter_reads:
            cfg = dict(cfg_base, **({"filter_read": fr} if fr else {}))
            label = fr or "default"
            warm = f"SELECT {a.id_col}, _distance FROM {alias} ORDER BY _distance LIMIT {a.k}"
            ann_sql(tbl, a.column, queries[0], warm, k=a.k, nprobe=a.nprobe, **cfg).to_pydict()
            for name, cond in cases:
                scan = tbl.scan(row_filter=cond) if cond else tbl.scan()
                n_files = len(list(scan.plan_files()))
                where = f"WHERE {cond}" if cond else ""
                for qi, q in enumerate(queries):
                    sql = f"SELECT {a.id_col}, _distance FROM {alias} {where} ORDER BY _distance LIMIT {a.k}"
                    t0 = time.perf_counter()
                    ids = ann_sql(tbl, a.column, q, sql, k=a.k, nprobe=a.nprobe, **cfg).to_pydict()[a.id_col]
                    dt = time.perf_counter() - t0
                    want = gt[(name, qi)] if gt else None   # may hold < k rows for rare filters
                    rec = len(set(ids) & want) / len(want) if want else None
                    rows.append({"runner": runner, "table": tname, "variant": label, "case": name,
                                 "q": qi, "s": dt, "recall": rec, "files": n_files, "ids": ids})
                    print(f"  {runner:6} {tname:20} {label:8} {name:16} q{qi} {dt:7.2f}s  "
                          f"files {n_files:4}" + (f"  recall {rec:.2f} of {len(want)}" if want else ""), flush=True)
    out = {"rows": rows}
    if runner == "ray":
        out["placement"] = placement()
        print("  work by node:", json.dumps(out["placement"]))
    return out


def placement():
    """Which nodes ran the work: alive actors and finished tasks per node IP."""
    import ray
    from ray.util.state import list_actors, list_nodes, list_tasks
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)
    ip = {n.node_id: n.node_ip for n in list_nodes()}
    by = {}
    for x in list_actors(filters=[("state", "=", "ALIVE")], limit=10000):
        by.setdefault(ip.get(x.node_id, x.node_id), {"actors": 0, "tasks": 0})["actors"] += 1
    for t in list_tasks(limit=10000):
        if t.node_id:
            by.setdefault(ip.get(t.node_id, t.node_id), {"actors": 0, "tasks": 0})["tasks"] += 1
    return by


def summarize(rows):
    keys = sorted({(r["runner"], r["table"], r["variant"]) for r in rows})
    cases = list(dict.fromkeys(r["case"] for r in rows))
    print(f"\n{'case':16} " + " ".join(f"{'/'.join(k):>34}" for k in keys))
    for c in cases:
        cells = []
        for k in keys:
            rs = [r for r in rows if (r["runner"], r["table"], r["variant"]) == k and r["case"] == c]
            rec = [r["recall"] for r in rs if r["recall"] is not None]
            cells.append(f"{statistics.median(r['s'] for r in rs):6.1f}s "
                         + (f"R@k {np.mean(rec):4.2f} " if rec else "") + f"{rs[0]['files']:4} files")
        print(f"{c:16} " + " ".join(f"{x:>34}" for x in cells))
    runners = sorted({r["runner"] for r in rows})
    if len(runners) == 2:                     # identical top-k across runners?
        same = total = 0
        for r in (x for x in rows if x["runner"] == runners[0]):
            o = [x for x in rows if x["runner"] == runners[1] and
                 (x["table"], x["variant"], x["case"], x["q"]) == (r["table"], r["variant"], r["case"], r["q"])]
            if o:
                total += 1
                same += o[0]["ids"] == r["ids"]
        print(f"\n{runners[0]} vs {runners[1]}: identical top-k in {same}/{total} queries")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="default")
    ap.add_argument("--table", action="append", required=True)
    ap.add_argument("--column", default="embedding")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--queries", required=True, help=".npy of query vectors")
    ap.add_argument("--n-queries", type=int, default=3)
    ap.add_argument("--where", action="append", help="'name=<WHERE clause>' (repeatable)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--runner", choices=("ray", "native", "both"), default="ray")
    ap.add_argument("--address", default=None, help="Ray address (default: auto/local)")
    ap.add_argument("--gt", action="store_true")
    ap.add_argument("--gt-cache", default=None)
    ap.add_argument("--filter-reads", nargs="+", default=[None])
    ap.add_argument("--cfg", action="append", default=[], help="SearchConfig override key=value")
    ap.add_argument("--out", default="ray_bench.json")
    ap.add_argument("--pass-runner", choices=("ray", "native"), help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.pass_runner:                         # child process for --runner both
        json.dump(run_pass(a, a.pass_runner), open(a.out, "w"))
        return
    if a.runner != "both":
        res = run_pass(a, a.runner)
        json.dump(res, open(a.out, "w"))
        summarize(res["rows"])
        return
    rows = []
    argv, skip = [], False
    for x in sys.argv[1:]:                    # drop --runner/--out; the child gets its own
        if skip:
            skip = False
            continue
        if x in ("--runner", "--out"):
            skip = True
            continue
        argv.append(x)
    for r in ("native", "ray"):
        out = f"{a.out}.{r}"
        subprocess.run([sys.executable, __file__, *argv, "--pass-runner", r, "--out", out], check=True)
        rows += json.load(open(out))["rows"]
    json.dump({"rows": rows}, open(a.out, "w"))
    summarize(rows)


if __name__ == "__main__":
    main()
