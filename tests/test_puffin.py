"""Per-file indexes in Puffin files: format, registry, attach, safety, front ends.

Local only (SQLite catalog, temp dirs, synthetic vectors). What is pinned down:

  1. format      our Puffin files parse with pyiceberg's own Puffin reader
  2. attach      indexing an existing table rewrites no data file, and every
                 front end then searches with the Puffin index, exactly at full probe
  3. ranged runs uncompressed embedding columns get IFVR runs from the file bytes
  4. idempotent  attaching twice indexes nothing new
  5. safety      a ref whose data file no longer matches (size/rows) is ignored:
                 exact search, still correct
  6. mixed       footer-indexed, Puffin-indexed and unindexed files in one table
  7. write path  write_indexed_table defaults to Puffin; footer location still works
  8. loadable    the table (with its registry property) still loads in pyiceberg
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lakehouse_ann.build import embedding_matrix, index_table      # noqa: E402
from lakehouse_ann.iceberg import STATS, create_table_from_files, vector_search  # noqa: E402
from lakehouse_ann.puffin import (IVF_BLOB, RUNS_BLOB, build_puffin, index_refs,  # noqa: E402
                                  read_footer, registered_puffins, registry_key)
from lakehouse_ann.write import attach_indexes, write_indexed_table  # noqa: E402

DIM = 16
FULL = 10_000                     # nprobe that covers every cluster -> exact answers


def _table(n, seed):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((10, DIM)).astype(np.float32) * 4
    v = centers[rng.integers(0, 10, n)] + rng.standard_normal((n, DIM)).astype(np.float32)
    return pa.table({
        "id": np.arange(n, dtype=np.int64) + seed * 100_000,
        "category": rng.choice(["tech", "news", "sports"], n),
        "embedding": pa.FixedSizeListArray.from_arrays(pa.array(v.ravel()), DIM),
    })


def _exact(src, q, mask=None, k=10):
    d = np.linalg.norm(embedding_matrix(src["embedding"]) - q, axis=1)
    if mask is not None:
        d[~mask] = np.inf
    return set(src["id"].to_numpy()[np.argsort(d)[:k]].tolist())


@pytest.fixture()
def cat(tmp_path):
    from pyiceberg.catalog.sql import SqlCatalog
    return SqlCatalog("t", uri=f"sqlite:///{tmp_path}/cat.db", warehouse=f"file://{tmp_path}/wh")


@pytest.fixture()
def plain_table(cat, tmp_path):
    """A table of plain Parquet files 'written by another engine': no index,
    snappy embeddings (file 0, 1) and an uncompressed one (file 2)."""
    d = tmp_path / "plain"
    d.mkdir()
    tables, paths = [_table(2000, s) for s in range(3)], []
    for i, t in enumerate(tables):
        p = str(d / f"part-{i}.parquet")
        if i < 2:
            pq.write_table(t, p, row_group_size=500)
        else:
            pq.write_table(t, p, row_group_size=500, compression={"embedding.list.element": "none",
                                                                 "embedding": "none"},
                           use_dictionary=["id", "category"], column_encoding={"embedding": "PLAIN"})
        paths.append(p)
    tbl = create_table_from_files(cat, "vec.plain", paths)
    return tbl, pa.concat_tables(tables), paths


def test_format_parses_with_pyiceberg():
    from pyiceberg.table.puffin import PuffinFile
    data = build_puffin([{"data_file": "f1", "size": 1, "rows": 2, "ivf": b"abc", "runs": b"xyz"}],
                        field_id=3, snapshot_id=7, sequence_number=1)
    pf = PuffinFile(data)
    assert [b.type for b in pf.footer.blobs] == [IVF_BLOB, RUNS_BLOB]
    assert pf.get_blob_payload(pf.footer.blobs[1]) == b"xyz"
    assert pf.footer.blobs[0].properties["referenced-data-file"] == "f1"


def test_attach_indexes_in_place_and_searches_exactly(cat, plain_table):
    tbl, src, paths = plain_table
    before = {p: open(p, "rb").read() for p in paths}
    entries, skipped = attach_indexes(tbl, "embedding", n_clusters=8, log=lambda *a: None)
    assert len(entries) == 3 and not skipped
    assert {p: open(p, "rb").read() for p in paths} == before          # no data file touched
    assert sum(1 for e in entries if e["runs"]) == 1                   # only the uncompressed file
    tbl = cat.load_table("vec.plain")                                  # still loads (8)
    assert len(registered_puffins(tbl, "embedding")) == 1
    assert len(index_refs(tbl, "embedding")) == 3
    blobs = read_footer(tbl.io, registered_puffins(tbl, "embedding")[0])
    assert {b["type"] for b in blobs} == {IVF_BLOB, RUNS_BLOB}

    q = embedding_matrix(src["embedding"])[42] + 0.05
    mask = src["category"].to_numpy(zero_copy_only=False) == "news"
    STATS.clear()
    got = vector_search(tbl, "embedding", q, k=10, row_filter="category = 'news'",
                        nprobe=FULL, selected_fields=("id",))
    assert set(got["id"].to_pylist()) == _exact(src, q, mask)
    assert {s.index_source for s in STATS} == {"puffin"}
    assert all(s.plan.startswith(("ivf", "exact:filtered")) for s in STATS)


def test_daft_and_sql_use_puffin(cat, plain_table):
    pytest.importorskip("daft")
    import daft
    from lakehouse_ann import ann_sql, read_ivf_iceberg
    tbl, src, _ = plain_table
    attach_indexes(tbl, "embedding", n_clusters=8, log=lambda *a: None)
    tbl = cat.load_table("vec.plain")
    q = embedding_matrix(src["embedding"])[7] + 0.05
    want = _exact(src, q)
    STATS.clear()
    df = read_ivf_iceberg(tbl, "embedding", q, k=10, nprobe=FULL).sort("_distance").limit(10)
    assert set(df.to_pydict()["id"]) == want
    assert {s.index_source for s in STATS} == {"puffin"}
    out = ann_sql(tbl, "embedding", q, k=10, nprobe=FULL,
                  sql="SELECT id FROM plain ORDER BY _distance LIMIT 10").to_pydict()
    assert set(out["id"]) == want


def test_attach_is_idempotent(cat, plain_table):
    tbl, _, _ = plain_table
    attach_indexes(tbl, "embedding", n_clusters=8, log=lambda *a: None)
    tbl = cat.load_table("vec.plain")
    entries, _ = attach_indexes(tbl, "embedding", n_clusters=8, log=lambda *a: None)
    assert entries == [] and len(registered_puffins(cat.load_table("vec.plain"), "embedding")) == 1


def test_mismatched_ref_is_ignored(cat, plain_table, tmp_path):
    """A registry entry that names a data file but with the wrong size/rows (a
    file rewritten in place, or a bad entry) must not be used."""
    from lakehouse_ann.puffin import write_puffin
    from lakehouse_ann.build import index_existing_file
    tbl, src, paths = plain_table
    ivf, _, rows = index_existing_file(tbl.io, paths[0], "embedding", 8)
    bad = str(tmp_path / "bad.puffin")
    write_puffin(tbl.io, bad, [{"data_file": paths[0], "size": 1, "rows": rows, "ivf": ivf}],
                 field_id=3, snapshot_id=1, sequence_number=1)
    with tbl.transaction() as tx:
        tx.set_properties({registry_key("embedding"): f'["{bad}"]'})
    tbl = cat.load_table("vec.plain")
    assert len(index_refs(tbl, "embedding")) == 1
    q = embedding_matrix(src["embedding"])[3] + 0.05
    STATS.clear()
    got = vector_search(tbl, "embedding", q, k=10, nprobe=1, selected_fields=("id",))
    assert set(got["id"].to_pylist()) == _exact(src, q)              # exact everywhere
    assert {s.index_source for s in STATS} == {""}
    assert {s.plan for s in STATS} == {"exact:no-index"}


def test_mixed_footer_puffin_and_unindexed(cat, tmp_path):
    d = tmp_path / "mixed"
    d.mkdir()
    tables = [_table(2000, s) for s in range(3)]
    paths = [str(d / f"f{i}.parquet") for i in range(3)]
    index_table(tables[0], paths[0], "embedding", 8, row_group_size=500)   # footer index
    pq.write_table(tables[1], paths[1])                                     # gets Puffin
    pq.write_table(tables[2], paths[2])                                     # stays unindexed
    tbl = create_table_from_files(cat, "vec.mixed", paths)
    from lakehouse_ann.write import attach_one, _write_puffins
    from lakehouse_ann.puffin import register
    f1 = next(t.file for t in tbl.scan().plan_files() if t.file.file_path == paths[1])
    e = attach_one(tbl.io, f1.file_path, f1.file_size_in_bytes, f1.record_count, "embedding", 8)
    with tbl.transaction() as tx:
        register(tbl, "embedding", _write_puffins(tbl, "embedding", [e]), tx)
    tbl = cat.load_table("vec.mixed")
    src = pa.concat_tables(tables)
    q = embedding_matrix(src["embedding"])[2500] + 0.05
    STATS.clear()
    got = vector_search(tbl, "embedding", q, k=10, nprobe=FULL, selected_fields=("id",))
    assert set(got["id"].to_pylist()) == _exact(src, q)
    by_file = {os.path.basename(s.path): s.index_source for s in STATS}
    assert by_file == {"f0.parquet": "footer", "f1.parquet": "puffin", "f2.parquet": ""}


def test_write_path_defaults_to_puffin_and_footer_still_works(cat, tmp_path):
    d = tmp_path / "staging"
    d.mkdir()
    src = _table(3000, 9)
    pq.write_table(src, d / "s.parquet")
    q = embedding_matrix(src["embedding"])[11] + 0.05
    for name, loc in (("vec.p", "puffin"), ("vec.f", "footer")):
        t = write_indexed_table(cat, name, str(d / "*.parquet"), "embedding", n_clusters=8,
                                partition_by=["category"], index_location=loc, log=lambda *a: None)
        files = [x.file.file_path for x in t.scan().plan_files()]
        has_footer = [b"ifs.vector.index.v1" in (pq.read_metadata(f).metadata or {}) for f in files]
        assert all(has_footer) == (loc == "footer") and any(has_footer) == (loc == "footer")
        assert bool(registered_puffins(t, "embedding")) == (loc == "puffin")
        STATS.clear()
        got = vector_search(t, "embedding", q, k=10, nprobe=FULL, selected_fields=("id",))
        assert set(got["id"].to_pylist()) == _exact(src, q)
        assert {s.index_source for s in STATS} == {loc}
