"""Write path: Parquet → indexed, partitioned, registered Iceberg table.

Local only (SQLite catalog, temp dirs, synthetic vectors; no model, no network).
What is pinned down:

  1. partition transforms   identity / truncate parsing and Iceberg semantics
  2. split + register       every output file holds one partition value; add_files
                            infers the spec; rows are preserved exactly
  3. pruning                a predicate on the partition columns plans only the
                            matching files
  4. chunking               inputs larger than max_rows_per_file become several files
  5. resume                 re-running uploads nothing and registers nothing new
  6. spec guard             an existing table keeps its spec; a different one raises
  7. search                 the result is searchable by every front end and exact
                            with a full probe

Run from the repo root:  .venv/bin/pytest -q tests/test_write.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lakehouse_ann.build import embedding_matrix                     # noqa: E402
from lakehouse_ann.iceberg import vector_search                      # noqa: E402
from lakehouse_ann.write import PartitionField, write_indexed_table  # noqa: E402

DIM = 16


def _table(n, seed):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((10, DIM)).astype(np.float32) * 4
    v = centers[rng.integers(0, 10, n)] + rng.standard_normal((n, DIM)).astype(np.float32)
    return pa.table({
        "id": np.arange(n, dtype=np.int64) + seed * 100_000,
        "customer_id": rng.integers(0, 400, n).astype(np.int64),
        "category": rng.choice(["tech", "news", "sports"], n),
        "embedding": pa.FixedSizeListArray.from_arrays(pa.array(v.ravel()), DIM),
    })


@pytest.fixture()
def env(tmp_path):
    from pyiceberg.catalog.sql import SqlCatalog
    src = tmp_path / "staging"
    src.mkdir()
    tables = [_table(3000, s) for s in range(3)]
    for i, t in enumerate(tables):
        pq.write_table(t, src / f"part-{i}.parquet")
    cat = SqlCatalog("t", uri=f"sqlite:///{tmp_path}/cat.db", warehouse=f"file://{tmp_path}/wh")
    return cat, str(src / "*.parquet"), pa.concat_tables(tables)


def _write(cat, glob, **kw):
    kw.setdefault("n_clusters", 16)
    return write_indexed_table(cat, "vec.docs", glob, "embedding", log=lambda *a: None, **kw)


def test_partition_field_parse_and_apply():
    assert PartitionField.parse("category") == PartitionField("category")
    f = PartitionField.parse("customer_id:truncate[200]")
    assert f == PartitionField("customer_id", 200) and f.name == "customer_id_trunc"
    assert [f.apply(v) for v in (0, 199, 200, 399, -1)] == [0, 0, 200, 200, -200]
    assert PartitionField("s", 2).apply("abcdef") == "ab"
    with pytest.raises(ValueError):
        PartitionField.parse("customer_id:bucket[4]")


def test_split_register_prune_and_rows(env):
    cat, glob, src = env
    t = _write(cat, glob, partition_by=["category", "customer_id:truncate[200]"])
    files = list(t.scan().plan_files())
    assert len(files) == 3 * 3 * 2                    # 3 inputs x 3 categories x 2 bands
    got = t.scan(selected_fields=("id",)).to_arrow()["id"].to_numpy()
    assert np.array_equal(np.sort(got), np.sort(src["id"].to_numpy()))
    for f in files:                                    # one partition value per file
        part = pq.read_table(f.file.file_path, columns=["category", "customer_id"])
        assert len(set(part["category"].to_pylist())) == 1
        assert len({c // 200 for c in part["customer_id"].to_pylist()}) == 1
    pruned = list(t.scan(row_filter="category = 'tech' and customer_id < 200").plan_files())
    assert len(pruned) == 3


def test_chunking_and_resume(env):
    cat, glob, src = env
    t = _write(cat, glob, max_rows_per_file=1000)
    files = list(t.scan().plan_files())
    assert len(files) == 9 and max(f.file.record_count for f in files) <= 1000
    snap = t.metadata.current_snapshot_id
    t2 = _write(cat, glob, max_rows_per_file=1000)
    assert t2.metadata.current_snapshot_id == snap    # nothing new registered
    assert sum(f.file.record_count for f in t2.scan().plan_files()) == src.num_rows


def test_existing_spec_is_kept_and_guarded(env):
    cat, glob, _ = env
    _write(cat, glob, partition_by=["category"])
    t = _write(cat, glob)                              # no partition_by: reuse the table's
    assert [f.name for f in t.spec().fields] == ["category"]
    with pytest.raises(ValueError, match="is partitioned"):
        _write(cat, glob, partition_by=["customer_id:truncate[200]"])


def test_written_table_searches_exactly_with_full_probe(env):
    cat, glob, src = env
    t = _write(cat, glob, partition_by=["category"])
    E = embedding_matrix(src["embedding"])
    q = E[42] + 0.05
    mask = src["category"].to_numpy(zero_copy_only=False) == "news"
    d = np.linalg.norm(E - q, axis=1)
    d[~mask] = np.inf
    want = set(src["id"].to_numpy()[np.argsort(d)[:10]])
    got = vector_search(t, "embedding", q, k=10, row_filter="category = 'news'",
                        nprobe=10_000, selected_fields=("id",))
    assert set(got["id"].to_pylist()) == want
