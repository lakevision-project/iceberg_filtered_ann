"""Tests for the SQL front end: SQL over Iceberg-planned, Daft-executed filtered ANN.

Skipped unless the demo catalog from bench/iceberg_demo.py exists
(data/iceberg/catalog.db); building it takes a couple of minutes.
That catalog's state matters: vec.docs_by_region's current snapshot has
the r3 file rewritten by a copy-on-write delete (no index), and its first
snapshot is all-indexed — which is exactly what the pruning, fallback and
time-travel tests exercise.

Run from the repo root:  .venv/bin/pytest -q tests/test_sql.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore", message=".*statistics missing.*")

daft = pytest.importorskip("daft")

from lakehouse_ann.build import embedding_matrix                # noqa: E402
from lakehouse_ann.iceberg import STATS, vector_search          # noqa: E402
from lakehouse_ann.iceberg_sql import ann_sql, read_ivf_iceberg  # noqa: E402
from pyiceberg.catalog.sql import SqlCatalog               # noqa: E402

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_CATALOG_DB = os.path.join(_ROOT, "data", "iceberg", "catalog.db")


@pytest.fixture(scope="module")
def cat():
    if not os.path.exists(_CATALOG_DB):
        pytest.skip("demo catalog missing; run bench/iceberg_demo.py first")
    return SqlCatalog("demo", uri=f"sqlite:///{_CATALOG_DB}",
                      warehouse="file://" + os.path.join(_ROOT, "data", "iceberg", "warehouse"))


@pytest.fixture(scope="module")
def docs(cat):
    return cat.load_table("vec.docs")


@pytest.fixture(scope="module")
def by_region(cat):
    return cat.load_table("vec.docs_by_region")


@pytest.fixture(scope="module")
def q(docs):
    first = next(iter(docs.scan().plan_files())).file.file_path
    rng = np.random.default_rng(123)
    return (embedding_matrix(pq.read_table(first, columns=["emb"]).column("emb"))[0]
            + rng.standard_normal(128).astype(np.float32) * 0.5)


def truth(tbl, row_filter, q, k, snapshot_id=None):
    """Exact top-k ids among matching rows, straight through pyiceberg."""
    scan = tbl.scan(row_filter=row_filter, snapshot_id=snapshot_id) if row_filter \
        else tbl.scan(snapshot_id=snapshot_id)
    t = scan.to_arrow()
    d = np.linalg.norm(embedding_matrix(t.column("emb")) - q, axis=1)
    order = np.argsort(d, kind="stable")[:k]
    return t.column("id").to_pylist(), order


def recall(got_ids, tids, order):
    want = {tids[i] for i in order}
    return len(want & set(got_ids)) / len(want)


def run(tbl, q, sql, k=10, **kw):
    STATS.clear()
    out = ann_sql(tbl, "emb", q, sql, k=k, **kw).to_pydict()
    return out, STATS.copy()


SQL_TOPK = ("SELECT id, tier, region, _distance FROM {t} WHERE {f} "
            "ORDER BY _distance LIMIT {n}")


def test_recall_moderate_selectivity(docs, q):
    out, st = run(docs, q, SQL_TOPK.format(t="docs", f="tier = 'p10'", n=10))
    tids, order = truth(docs, "tier = 'p10'", q, 10)
    assert len(out["id"]) == 10
    assert recall(out["id"], tids, order) >= 0.99
    assert len(st) == 16                       # random layout: nothing pruned
    assert all(s.n_match is not None for s in st)   # filter reached every kernel


def test_parity_with_pyiceberg_frontend(docs, q):
    """Same kernel, same plan: SQL and vector_search() must agree exactly."""
    for flt in ("tier = 'p10'", "region = 'r3'", "region = 'r3' and tier = 'p10'"):
        out, _ = run(docs, q, SQL_TOPK.format(t="docs", f=flt, n=10))
        ref = vector_search(docs, "emb", q, k=10, row_filter=flt,
                            selected_fields=("id",))
        assert out["id"] == ref.column("id").to_pylist(), flt


def test_partition_pruning_and_no_index_fallback(by_region, q):
    """SQL WHERE -> pyiceberg planning prunes to the one r3 file; it is the
    copy-on-write rewrite (no index), searched exactly, still correct."""
    out, st = run(by_region, q, SQL_TOPK.format(t="docs_by_region", f="region = 'r3'", n=10))
    tids, order = truth(by_region, "region = 'r3'", q, 10)
    assert len(st) == 1
    assert st[0].plan == "exact:no-index"
    assert recall(out["id"], tids, order) >= 0.99


def test_time_travel(by_region, q):
    snap0 = by_region.metadata.snapshots[0].snapshot_id
    out, st = run(by_region, q, SQL_TOPK.format(t="docs_by_region", f="region = 'r3'", n=10),
                  snapshot_id=snap0)
    tids, order = truth(by_region, "region = 'r3'", q, 10, snapshot_id=snap0)
    assert len(st) == 1
    assert st[0].plan == "ivf"                 # pure file: residual AlwaysTrue
    assert st[0].n_match is None               # no filter column read at all
    assert recall(out["id"], tids, order) >= 0.9


def test_join_over_ann_result(docs, q):
    meta = daft.from_pydict({"region": ["r3", "r7"], "label": ["eu", "us"]})
    out, st = run(docs, q, """
        SELECT a.id, m.label, a._distance FROM docs a JOIN meta m ON a.region = m.region
        WHERE a.tier = 'p1' ORDER BY a._distance LIMIT 10""", bindings={"meta": meta})
    assert len(out["id"]) == 10
    assert set(out["label"]) <= {"eu", "us"}
    assert any(s.n_match is not None for s in st)   # WHERE pushed under the join


def test_aggregation_over_topk_subquery(docs, q):
    out, _ = run(docs, q, """
        SELECT region, count(*) AS n FROM (
            SELECT region, _distance FROM docs WHERE tier = 'p1'
            ORDER BY _distance LIMIT 10) GROUP BY region""")
    assert sum(out["n"]) == 10


def test_no_match_filter_skips_index(docs, q):
    out, st = run(docs, q, SQL_TOPK.format(t="docs", f="tier = 'nope'", n=10))
    assert out["id"] == []
    assert all(s.plan == "skip:no-match" for s in st)


def test_distance_filter_applied_after_ranking(docs, q):
    out, _ = run(docs, q, "SELECT id, _distance FROM docs WHERE _distance < 1e9 "
                          "ORDER BY _distance LIMIT 10")
    assert len(out["id"]) == 10                # correct, just not pre-filtered


def test_limit_larger_than_k_is_raised(docs, q):
    out, _ = run(docs, q, SQL_TOPK.format(t="docs", f="tier = 'p10'", n=15), k=10)
    tids, order = truth(docs, "tier = 'p10'", q, 15)
    assert len(out["id"]) == 15
    assert recall(out["id"], tids, order) >= 0.99


def test_post_filter_trap_is_real(docs, q):
    """A WHERE after ORDER BY..LIMIT cannot be pushed: fewer than k rows.
    This documents the operator-order rule, not a bug."""
    out, _ = run(docs, q, """
        SELECT id FROM (SELECT id, tier, _distance FROM docs
                        ORDER BY _distance LIMIT 10) WHERE tier = 'p1'""")
    assert len(out["id"]) < 10


def test_dataframe_api_matches_sql(docs, q):
    df = read_ivf_iceberg(docs, "emb", q, k=10) \
        .where(daft.col("tier") == "p10").sort("_distance").limit(10)
    ref = vector_search(docs, "emb", q, k=10, row_filter="tier = 'p10'",
                        selected_fields=("id",))
    assert df.to_pydict()["id"] == ref.column("id").to_pylist()
